use std::sync::atomic::AtomicI64;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use log::{info, trace, warn};
use serde::Serialize;
use tokio::sync::mpsc::{Receiver, Sender, UnboundedSender};
use tokio::sync::{broadcast, Mutex};
use tokio::task::JoinHandle;

use crate::requests::{
    TextGenerationAggregatedResponse, TextGenerationBackend, TextGenerationRequest,
    TextRequestGenerator,
};

#[serde_with::serde_as]
#[derive(Clone, Serialize)]
pub struct ExecutorConfig {
    pub max_vus: u64,
    #[serde(rename = "duration_secs")]
    #[serde_as(as = "serde_with::DurationSeconds<u64>")]
    pub duration: Duration,
    pub rate: Option<f64>,
}

#[async_trait]
pub trait Executor {
    async fn run(
        &self,
        requests: Arc<Mutex<dyn TextRequestGenerator + Send>>,
        responses_tx: UnboundedSender<TextGenerationAggregatedResponse>,
        stop_sender: broadcast::Sender<()>,
    );
}

pub struct ConstantVUsExecutor {
    config: ExecutorConfig,
    backend: Box<dyn TextGenerationBackend + Send + Sync>,
}

impl ConstantVUsExecutor {
    pub fn new(
        backend: Box<dyn TextGenerationBackend + Send + Sync>,
        max_vus: u64,
        duration: Duration,
    ) -> ConstantVUsExecutor {
        Self {
            backend,
            config: ExecutorConfig {
                max_vus,
                duration,
                rate: None,
            },
        }
    }
}

#[async_trait]
impl Executor for ConstantVUsExecutor {
    async fn run(
        &self,
        requests: Arc<Mutex<dyn TextRequestGenerator + Send>>,
        responses_tx: UnboundedSender<TextGenerationAggregatedResponse>,
        stop_sender: broadcast::Sender<()>,
    ) {
        let start = std::time::Instant::now();
        let mut set = tokio::task::JoinSet::new();

        // start all VUs as long-lived tasks
        for _ in 0..self.config.max_vus {
            let backend = self.backend.clone();
            let responses_tx = responses_tx.clone();
            let mut stop_receiver = stop_sender.subscribe();
            let requests = requests.clone();
            let duration = self.config.duration;

            set.spawn(async move {
                loop {
                    // Check for stop signal or duration
                    if start.elapsed() > duration {
                        break;
                    }
                    // Check stop signal non-blocking
                    if stop_receiver.try_recv().is_ok() {
                        break;
                    }

                    // Generate request
                    let mut requests_guard = requests.lock().await;
                    let request = Arc::from(requests_guard.generate_request());
                    drop(requests_guard);

                    // Execute request
                    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
                    trace!("VU started request: {:?}", request);
                    
                    // We run generation in a separate task to match backend interface which might block?
                    // Actually Backend::generate is async. We can just await it if we want sequential execution within a VU.
                    // The original start_vu spawned a thread for backend.generate(..., tx).
                    // Let's stick to the pattern but without the overhead if possible.
                    // But backend.generate takes a Sender.
                    let backend_clone = backend.clone();
                    let request_clone = request.clone();
                    
                    let gen_join = tokio::spawn(async move {
                        backend_clone.generate(request_clone, tx).await;
                    });

                    // Forward responses
                    while let Some(response) = rx.recv().await {
                         if responses_tx.send(response).is_err() {
                             // channel closed
                             return;
                         }
                    }
                    
                    let _ = gen_join.await;
                }
                // Signal ended (one per VU, but we can just do one at the very end or not at all? 
                // The original code sent new_as_ended() when duration reached.
                // We should probably do that once per VU when it exits?
                // Or just rely on the scheduler counting?
                // The original code had: responses_tx.send(TextGenerationAggregatedResponse::new_as_ended());
                // Let's send it when VU finishes naturally.
                let _ = responses_tx.send(TextGenerationAggregatedResponse::new_as_ended());
            });
        }

        // Wait for all VUs to finish
        let mut stop_receiver = stop_sender.subscribe();
        
        // We need to wait for either all tasks to join OR a stop signal.
        loop {
            tokio::select! {
                _ = stop_receiver.recv() => {
                    // Stop signal received. The VUs check this signal in their loop, 
                    // but they might be stuck in backend.generate.
                    // We can abort them.
                    set.shutdown().await;
                    return;
                },
                join_next = set.join_next() => {
                    match join_next {
                        Some(_) => {}, // A VU finished
                        None => {
                            // All VUs finished
                            return;
                        }
                    }
                }
            }
        }
    }
}

async fn start_vu(
    backend: Box<dyn TextGenerationBackend + Send + Sync>,
    request: Arc<TextGenerationRequest>,
    responses_tx: UnboundedSender<TextGenerationAggregatedResponse>,
    end_tx: Sender<bool>,
    stop_sender: broadcast::Sender<()>,
) -> JoinHandle<()> {
    let mut stop_receiver = stop_sender.subscribe();
    tokio::spawn(async move {
        tokio::select! {
            _ = stop_receiver.recv() => {
                let _ = end_tx.send(true).await;
            },
            _ = async{
                let (tx, mut rx): (Sender<TextGenerationAggregatedResponse>, Receiver<TextGenerationAggregatedResponse>) = tokio::sync::mpsc::channel(1);
                trace!("VU started with request: {:?}", request);
                let req_thread = tokio::spawn(async move {
                    backend.generate(request.clone(), tx).await;
                });
                let send_thread = tokio::spawn(async move {
                    while let Some(response) = rx.recv().await {
                        // ignore errors, if the receiver is gone we want to finish the request
                        // to leave remote server in clean state
                        let _ = responses_tx.send(response);
                    }
                });
                req_thread.await.unwrap();
                send_thread.await.unwrap();
                // signal that the VU work is done
                let _ = end_tx.send(true).await;
            }=>{}
        }
    })
}

pub struct ConstantArrivalRateExecutor {
    config: ExecutorConfig,
    backend: Box<dyn TextGenerationBackend + Send + Sync>,
}

impl ConstantArrivalRateExecutor {
    pub fn new(
        backend: Box<dyn TextGenerationBackend + Send + Sync>,
        max_vus: u64,
        duration: Duration,
        rate: f64,
    ) -> ConstantArrivalRateExecutor {
        Self {
            backend,
            config: ExecutorConfig {
                max_vus,
                duration,
                rate: Some(rate),
            },
        }
    }
}

#[async_trait]
impl Executor for ConstantArrivalRateExecutor {
    async fn run(
        &self,
        requests: Arc<Mutex<dyn TextRequestGenerator + Send>>,
        responses_tx: UnboundedSender<TextGenerationAggregatedResponse>,
        stop_sender: broadcast::Sender<()>,
    ) {
        let start = std::time::Instant::now();
        let active_vus = Arc::new(AtomicI64::new(0));
        // channel to handle ending VUs
        let (end_tx, mut end_rx): (Sender<bool>, Receiver<bool>) =
            tokio::sync::mpsc::channel(self.config.max_vus as usize);
        let rate = self.config.rate.expect("checked in new()");
        // spawn new VUs every `tick_ms` to reach the expected `rate` per second, until the duration is reached
        let tick_ms = 10;
        let mut interval = tokio::time::interval(Duration::from_millis(tick_ms));

        let backend = self.backend.clone();
        let duration = self.config.duration;
        let max_vus = self.config.max_vus;
        let active_vus_thread = active_vus.clone();
        let mut stop_receiver_signal = stop_sender.subscribe();
        let vu_thread = tokio::spawn(async move {
            tokio::select! {
                _ = stop_receiver_signal.recv() => {},
                _= async {
                    let mut spawn_queue = 0.; // start with at least one VU
                    while start.elapsed() < duration {
                        spawn_queue += rate * (tick_ms as f64) / 1000.0;
                        // delay spawning if we can't spawn a full VU yet
                        if spawn_queue < 1.0 {
                            interval.tick().await;
                            continue;
                        }
                        // spawn VUs, keep track of the fraction of VU to spawn for the next iteration
                        let to_spawn = spawn_queue.floor() as u64;
                        spawn_queue -= to_spawn as f64;
                        for _ in 0..to_spawn {
                            if active_vus_thread.load(std::sync::atomic::Ordering::SeqCst) < max_vus as i64 {
                                let mut requests_guard = requests.lock().await;
                                let request = Arc::from(requests_guard.generate_request());
                                start_vu(backend.clone(), request.clone(), responses_tx.clone(), end_tx.clone(),stop_sender.clone()).await;
                                active_vus_thread.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            } else {
                                warn!("Max VUs reached, skipping request");
                                break;
                            }
                        }
                        interval.tick().await;
                    }
                    // signal that the VU work is done
                    info!("Duration reached, waiting for all VUs to finish...");
                    let _ = responses_tx.send(TextGenerationAggregatedResponse::new_as_ended());
                }=>{}
            }
        });
        while end_rx.recv().await.is_some() {
            active_vus.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            // wait for all VUs to finish
            if start.elapsed() > self.config.duration
                && active_vus.load(std::sync::atomic::Ordering::SeqCst) == 0
            {
                break;
            }
        }
        // wait for the VU thread to finish
        vu_thread.await.unwrap();
    }
}
