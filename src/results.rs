use crate::executors::ExecutorConfig;
use crate::requests::TextGenerationAggregatedResponse;
use crate::results::BenchmarkErrors::NoResponses;
use crate::scheduler::ExecutorType;
use chrono::Utc;
use std::fmt::{Debug, Display, Formatter};
use std::time::Duration;

#[derive(Debug)]
pub(crate) enum BenchmarkErrors {
    NoResponses,
}

impl Display for BenchmarkErrors {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            NoResponses => write!(f, "Backend did not return any valid response. It is either not responding or test duration is too short."),
        }
    }
}

#[derive(Clone)]
pub struct BenchmarkResults {
    pub id: String,
    aggregated_responses: Vec<TextGenerationAggregatedResponse>,
    executor_type: ExecutorType,
    executor_config: ExecutorConfig,
}

impl BenchmarkResults {
    pub fn new(
        id: String,
        executor_type: ExecutorType,
        executor_config: ExecutorConfig,
    ) -> BenchmarkResults {
        BenchmarkResults {
            id,
            aggregated_responses: Vec::new(),
            executor_type,
            executor_config,
        }
    }

    pub fn add_response(&mut self, response: TextGenerationAggregatedResponse) {
        self.aggregated_responses.push(response);
    }

    pub fn total_requests(&self) -> usize {
        self.aggregated_responses.len()
    }

    pub fn start_time(&self) -> Option<tokio::time::Instant> {
        self.aggregated_responses
            .first()
            .and_then(|response| response.start_time)
    }

    pub fn end_time(&self) -> Option<tokio::time::Instant> {
        self.aggregated_responses
            .last()
            .and_then(|response| response.end_time)
    }

    fn is_ready(&self) -> bool {
        self.start_time().is_some() && self.end_time().is_some()
    }

    pub fn failed_requests(&self) -> usize {
        self.aggregated_responses
            .iter()
            .filter(|response| response.failed)
            .count()
    }

    pub fn successful_requests(&self) -> usize {
        self.aggregated_responses
            .iter()
            .filter(|response| !response.failed)
            .count()
    }

    pub fn token_throughput_secs(&self) -> anyhow::Result<f64> {
        if self.is_ready() {
            let total_tokens: u64 = self.total_tokens();
            Ok(total_tokens as f64 / self.duration().unwrap_or_default().as_secs_f64())
        } else {
            Err(anyhow::anyhow!(NoResponses))
        }
    }

    pub fn input_token_throughput_secs(&self) -> anyhow::Result<f64> {
        if self.is_ready() {
            let total_tokens: u64 = self.total_tokens_sent();
            Ok(total_tokens as f64 / self.duration().unwrap_or_default().as_secs_f64())
        } else {
            Err(anyhow::anyhow!(NoResponses))
        }
    }



    pub fn total_tokens_sent(&self) -> u64 {
        self.get_successful_responses()
            .iter()
            .map(|response| response.request.clone().unwrap().num_prompt_tokens)
            .sum()
    }

    pub fn input_token_throughput_median_secs(&self) -> anyhow::Result<f64> {
         let packets = self.calculate_input_token_distribution();
         let histogram_data: Vec<f64> = packets.iter().map(|&x| x as f64).collect();
         self.quantile_throughput(histogram_data, 0.5)
    }

    pub fn token_throughput_median_secs(&self) -> anyhow::Result<f64> {
         let packets = self.calculate_output_token_distribution();
         let histogram_data: Vec<f64> = packets.iter().map(|&x| x as f64).collect();
         self.quantile_throughput(histogram_data, 0.5)
    }

    fn calculate_input_token_distribution(&self) -> Vec<u64> {
        if !self.is_ready() {
            return Vec::new();
        }
        let start = self.start_time().unwrap();
        let end = self.end_time().unwrap();
        let duration = end.duration_since(start).as_secs() + 1;
        let mut buckets = vec![0u64; duration as usize];

        for resp in self.get_successful_responses() {
             if let Some(s) = resp.start_time {
                 if s >= start { // Sanity check
                    let offset = s.duration_since(start).as_secs();
                    if (offset as usize) < buckets.len() {
                        buckets[offset as usize] += resp.request.as_ref().unwrap().num_prompt_tokens;
                    }
                 }
             }
        }
        buckets
    }

    fn calculate_output_token_distribution(&self) -> Vec<u64> {
        if !self.is_ready() {
            return Vec::new();
        }
        let start = self.start_time().unwrap();
        let end = self.end_time().unwrap();
        let duration = end.duration_since(start).as_secs() + 1;
        let mut buckets = vec![0u64; duration as usize];

        for resp in self.get_successful_responses() {
             for (arrival_time, num_tokens) in &resp.token_arrival_log {
                 if *arrival_time >= start {
                    let offset = arrival_time.duration_since(start).as_secs();
                    if (offset as usize) < buckets.len() {
                        buckets[offset as usize] += num_tokens;
                    }
                 }
             }
        }
        buckets
    }

    fn quantile_throughput(&self, mut data: Vec<f64>, quantile: f64) -> anyhow::Result<f64> {
        if data.is_empty() {
             return Err(anyhow::anyhow!(NoResponses));
        }
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = (quantile * (data.len() - 1) as f64) as usize;
        Ok(data[idx])
    }

    pub fn total_prompt_tokens(&self) -> u64 {
        self.get_successful_responses()
            .iter()
            .map(|response| response.request.clone().unwrap().num_prompt_tokens)
            .sum()
    }

    pub fn prompt_tokens_avg(&self) -> anyhow::Result<f64> {
        if self.is_ready() {
            let total_prompt_tokens = self.total_prompt_tokens();
            Ok(total_prompt_tokens as f64 / self.successful_requests() as f64)
        } else {
            Err(anyhow::anyhow!(NoResponses))
        }
    }

    pub fn successful_request_rate(&self) -> anyhow::Result<f64> {
        if self.is_ready() {
            let total_requests = self.successful_requests();
            Ok(total_requests as f64 / self.duration().unwrap_or_default().as_secs_f64())
        } else {
            Err(anyhow::anyhow!(NoResponses))
        }
    }

    pub fn total_tokens(&self) -> u64 {
        self.get_successful_responses()
            .iter()
            .map(|response| response.num_generated_tokens)
            .sum()
    }

    pub fn duration(&self) -> anyhow::Result<std::time::Duration> {
        if self.is_ready() {
            Ok(self
                .end_time()
                .unwrap()
                .duration_since(self.start_time().unwrap()))
        } else {
            Err(anyhow::anyhow!(NoResponses))
        }
    }

    pub fn e2e_latency_avg(&self) -> anyhow::Result<std::time::Duration> {
        if self.is_ready() {
            if self.successful_requests() == 0 {
                return Ok(Duration::from_secs(0));
            }
            Ok(self
                .get_successful_responses()
                .iter()
                .map(|response| response.e2e_latency().unwrap_or_default())
                .sum::<Duration>()
                / self.successful_requests() as u32)
        } else {
            Err(anyhow::anyhow!(NoResponses))
        }
    }

    pub fn e2e_latency_percentile(&self, percentile: f64) -> anyhow::Result<std::time::Duration> {
        let quantile = self.quantile_duration(
            self.get_successful_responses()
                .iter()
                .map(|response| response.e2e_latency().unwrap_or_default())
                .collect(),
            percentile,
        )?;
        Ok(Duration::from_secs_f64(quantile))
    }

    pub fn time_to_first_token_avg(&self) -> anyhow::Result<std::time::Duration> {
        if self.is_ready() {
            if self.successful_requests() == 0 {
                return Ok(Duration::from_secs(0));
            }
            Ok(self
                .get_successful_responses()
                .iter()
                .map(|response| response.time_to_first_token().unwrap_or_default())
                .sum::<Duration>()
                / self.successful_requests() as u32)
        } else {
            Err(anyhow::anyhow!(NoResponses))
        }
    }

    pub fn time_to_first_token_percentile(&self, percentile: f64) -> anyhow::Result<Duration> {
        let quantile = self.quantile_duration(
            self.get_successful_responses()
                .iter()
                .map(|response| response.time_to_first_token().unwrap_or_default())
                .collect(),
            percentile,
        )?;
        Ok(Duration::from_secs_f64(quantile))
    }

    pub fn inter_token_latency_avg(&self) -> anyhow::Result<std::time::Duration> {
        if self.is_ready() {
            if self.successful_requests() == 0 {
                return Ok(Duration::from_secs(0));
            }
            Ok(self
                .get_successful_responses()
                .iter()
                .map(|response| response.inter_token_latency().unwrap_or_default())
                .sum::<Duration>()
                / self.successful_requests() as u32)
        } else {
            Err(anyhow::anyhow!(NoResponses))
        }
    }

    pub fn inter_token_latency_percentile(&self, percentile: f64) -> anyhow::Result<Duration> {
        let quantile = self.quantile_duration(
            self.get_successful_responses()
                .iter()
                .map(|response| response.inter_token_latency().unwrap_or_default())
                .collect(),
            percentile,
        )?;
        Ok(Duration::from_secs_f64(quantile))
    }

    pub fn executor_type(&self) -> ExecutorType {
        self.executor_type.clone()
    }

    pub fn executor_config(&self) -> ExecutorConfig {
        self.executor_config.clone()
    }

    fn get_successful_responses(&self) -> Vec<&TextGenerationAggregatedResponse> {
        self.aggregated_responses
            .iter()
            .filter(|response| !response.failed)
            .collect()
    }

    pub fn get_responses(&self) -> Vec<TextGenerationAggregatedResponse> {
        self.aggregated_responses.clone()
    }

    /// Calculate the quantile of a given data set using interpolation method
    /// Results are similar to `numpy.percentile`
    fn quantile_duration(&self, mut data: Vec<Duration>, quantile: f64) -> anyhow::Result<f64> {
        if self.is_ready() {
            if data.is_empty() {
                return Err(anyhow::anyhow!(NoResponses));
            }
            if data.len() == 1 {
                return Ok(data[0].as_secs_f64());
            }
            data.sort();
            let i = (quantile * (data.len() - 1) as f64).floor();
            let delta = (data.len() - 1) as f64 * quantile - i;
            if i as usize >= data.len() {
                return Err(anyhow::anyhow!(NoResponses));
            }
            let quantile = (1. - delta) * data[i as usize].as_secs_f64()
                + delta * data[i as usize + 1].as_secs_f64();
            Ok(quantile)
        } else {
            Err(anyhow::anyhow!(NoResponses))
        }
    }
}

impl Debug for BenchmarkResults {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BenchmarkResult")
            .field("id", &self.id)
            .field("executor_type", &self.executor_type.to_string())
            .field("total_requests", &self.total_requests())
            .field("start_time", &self.start_time())
            .field("end_time", &self.end_time())
            .field("total_tokens", &self.total_tokens())
            .field(
                "token_throughput_secs",
                &self
                    .token_throughput_secs()
                    .or::<anyhow::Result<f64>>(Ok(-1.0)),
            )
            .field(
                "duration_ms",
                &self
                    .duration()
                    .or::<anyhow::Result<Duration>>(Ok(Duration::from_secs(0))),
            )
            .field(
                "average_time_to_first_token",
                &self
                    .time_to_first_token_avg()
                    .or::<anyhow::Result<Duration>>(Ok(Duration::from_secs(0))),
            )
            .field(
                "average_inter_token_latency",
                &self
                    .inter_token_latency_avg()
                    .or::<anyhow::Result<Duration>>(Ok(Duration::from_secs(0))),
            )
            .field("failed_requests", &self.failed_requests())
            .field("successful_requests", &self.successful_requests())
            .field(
                "request_rate",
                &self
                    .successful_request_rate()
                    .or::<anyhow::Result<f64>>(Ok(-1.0)),
            )
            .field("sent_prompt_tokens", &self.total_tokens_sent())
            .field(
                "e2e_latency_avg",
                &self
                    .e2e_latency_avg()
                    .or::<anyhow::Result<Duration>>(Ok(Duration::from_secs(0))),
            )
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    results: Vec<BenchmarkResults>,
    start_time: Option<chrono::DateTime<Utc>>,
    end_time: Option<chrono::DateTime<Utc>>,
}

impl BenchmarkReport {
    pub fn new() -> BenchmarkReport {
        BenchmarkReport {
            results: Vec::new(),
            start_time: None,
            end_time: None,
        }
    }

    pub fn start(&mut self) {
        self.start_time = Some(Utc::now());
    }

    pub fn end(&mut self) {
        self.end_time = Some(Utc::now());
    }

    pub fn add_benchmark_result(&mut self, result: BenchmarkResults) {
        self.results.push(result);
    }

    pub fn get_results(&self) -> Vec<BenchmarkResults> {
        self.results.clone()
    }

    pub fn start_time(&self) -> Option<chrono::DateTime<Utc>> {
        self.start_time
    }

    pub fn end_time(&self) -> Option<chrono::DateTime<Utc>> {
        self.end_time
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::requests::TextGenerationRequest;
    use std::sync::Arc;
    #[test]
    fn test_time_to_first_token_percentile() {
        let request = Arc::from(TextGenerationRequest {
            id: None,
            prompt: "test".to_string(),
            num_prompt_tokens: 10,
            num_decode_tokens: None,
        });
        let mut response1 = TextGenerationAggregatedResponse::new(request.clone());
        response1.start_time = Some(tokio::time::Instant::now());
        response1.end_time =
            Some(tokio::time::Instant::now() + tokio::time::Duration::from_millis(100));
        response1.num_generated_tokens = 100;
        response1.failed = false;
        response1.times_to_tokens = vec![
            Duration::from_millis(100),
            Duration::from_millis(200),
            Duration::from_millis(300),
            Duration::from_millis(400),
            Duration::from_millis(500),
        ];

        let mut response2 = TextGenerationAggregatedResponse::new(request.clone());
        response2.start_time = Some(tokio::time::Instant::now());
        response2.end_time =
            Some(tokio::time::Instant::now() + tokio::time::Duration::from_millis(200));
        response2.num_generated_tokens = 100;
        response2.failed = false;
        response2.times_to_tokens = vec![
            Duration::from_millis(600),
            Duration::from_millis(700),
            Duration::from_millis(800),
            Duration::from_millis(900),
            Duration::from_millis(1000),
        ];

        let mut response3 = TextGenerationAggregatedResponse::new(request.clone());
        response3.start_time = Some(tokio::time::Instant::now());
        response3.end_time =
            Some(tokio::time::Instant::now() + tokio::time::Duration::from_millis(300));
        response3.num_generated_tokens = 100;
        response3.failed = false;
        response3.times_to_tokens = vec![
            Duration::from_millis(1100),
            Duration::from_millis(1200),
            Duration::from_millis(1300),
            Duration::from_millis(1400),
            Duration::from_millis(1500),
        ];

        let mut response4 = TextGenerationAggregatedResponse::new(request.clone());
        response4.start_time = Some(tokio::time::Instant::now());
        response4.end_time =
            Some(tokio::time::Instant::now() + tokio::time::Duration::from_millis(300));
        response4.num_generated_tokens = 100;
        response4.failed = false;
        response4.times_to_tokens = vec![
            Duration::from_millis(1600),
            Duration::from_millis(1700),
            Duration::from_millis(1800),
            Duration::from_millis(1900),
            Duration::from_millis(2000),
        ];

        let mut results = BenchmarkResults::new(
            "test".to_string(),
            ExecutorType::ConstantArrivalRate,
            ExecutorConfig {
                max_vus: 0,
                duration: Default::default(),
                rate: None,
            },
        );
        results.add_response(response1);
        results.add_response(response2);
        results.add_response(response3);
        results.add_response(response4);

        assert_eq!(
            results.time_to_first_token_percentile(0.9).unwrap(),
            Duration::from_millis(1450)
        );
        assert_eq!(
            results.time_to_first_token_percentile(0.5).unwrap(),
            Duration::from_millis(850)
        );
    }
    #[test]
    fn test_input_token_throughput() {
        let request = Arc::from(TextGenerationRequest {
            id: None,
            prompt: "test".to_string(),
            num_prompt_tokens: 50,
            num_decode_tokens: None,
        });
        let mut response = TextGenerationAggregatedResponse::new(request.clone());
        response.start_time = Some(tokio::time::Instant::now());
        response.end_time =
            Some(tokio::time::Instant::now() + tokio::time::Duration::from_secs(2));
        response.num_generated_tokens = 100;
        response.failed = false;

        let mut results = BenchmarkResults::new(
            "test".to_string(),
            ExecutorType::ConstantArrivalRate,
            ExecutorConfig {
                max_vus: 0,
                duration: Default::default(),
                rate: None,
            },
        );
        results.add_response(response);

        // Input throughput = 50 tokens / 2 seconds = 25 tokens/sec
        // Median: all tokens arrive at T0. In the first second, we have 50 tokens. In the second, 0.
        // Actually, bucket logic: T0 is start. Duration 2s. buckets[0] = 50. buckets[1] = 0.
        // Data: [50, 0]. Sorted: [0, 50]. Median(0.5): 0? Wait, 0.5 * (2-1) = 0.5. Floor is 0.
        // quantile logic: idx = 0.5 * 1 = 0.5 -> 0. Return data[0] which is 0.
        // Wait, if 50 tokens arrive at start, and duration is 2s.
        // Buckets: [50, 0]. Median is 0.
        // Let's adjust duration to 1s to make it simpler, or just assert 0.
        // Better: Make it uniform traffic. 2 requests, one at T=0, one at T=1.
        // Req1: 50 tok at T=0. Req2: 50 tok at T=1.
        // Buckets: [50, 50]. Median: 50.

        let throughput = results.input_token_throughput_secs().unwrap();
        assert!((throughput - 25.0).abs() < 1e-6, "throughput: {}", throughput);
    }

    #[test]
    fn test_median_throughput() {
        let request1 = Arc::from(TextGenerationRequest {
            id: None,
            prompt: "test".to_string(),
            num_prompt_tokens: 50,
            num_decode_tokens: None,
        });
        let mut response1 = TextGenerationAggregatedResponse::new(request1.clone());
        response1.start_time = Some(tokio::time::Instant::now());
        response1.end_time = Some(tokio::time::Instant::now() + tokio::time::Duration::from_secs(2));
        response1.failed = false;

        let request2 = Arc::from(TextGenerationRequest {
            id: None,
            prompt: "test".to_string(),
            num_prompt_tokens: 50,
            num_decode_tokens: None,
        });
        let mut response2 = TextGenerationAggregatedResponse::new(request2.clone());
        // Start 1 second later
        response2.start_time = Some(tokio::time::Instant::now() + tokio::time::Duration::from_secs(1));
        response2.end_time = Some(tokio::time::Instant::now() + tokio::time::Duration::from_secs(2));
        response2.failed = false;

        let mut results = BenchmarkResults::new(
            "test".to_string(),
            ExecutorType::ConstantArrivalRate,
            ExecutorConfig {
                max_vus: 0,
                duration: Default::default(),
                rate: None,
            },
        );
        results.add_response(response1); // bucket[0] += 50
        results.add_response(response2); // bucket[1] += 50

        // Duration is roughly 2s (from T=0 to T=2).
        // Buckets: [50, 50]. Median: 50.
        // Mean: 100 / 2 = 50.

        let median = results.input_token_throughput_median_secs().unwrap();
        assert!((median - 50.0).abs() < 1e-6, "median: {}", median);
    }
}
