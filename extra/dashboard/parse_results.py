import argparse
import json
import os

import pandas as pd


def build_df(model: str, data_files: dict[str, str]) -> pd.DataFrame:
    df = pd.DataFrame()
    # Load the results
    for key, filename in data_files.items():
        with open(filename, 'r') as f:
            data = json.load(f)
            if data['config']['meta'] is None:
                data['config']['meta'] = {}
            for result in data['results']:
                entry = pd.json_normalize(result).to_dict(orient='records')[0]
                if 'engine' in data['config']['meta']:
                    entry['engine'] = data['config']['meta']['engine']
                if 'tp' in data['config']['meta']:
                    entry['tp'] = data['config']['meta']['tp']
                if 'version' in data['config']['meta']:
                    entry['version'] = data['config']['meta']['version']
                if 'device' in data['config']['meta']:
                    entry['device'] = data['config']['meta']['device']
                entry['model'] = data['config']['model_name']
                entry['run_id'] = data['config']['run_id']

                # Extract system info
                if 'system' in data:
                    system = data['system']
                    cpu_info = system.get('cpu', ['Unknown'])[0] # Just one line
                    os_name = system.get('os_name', 'Unknown')
                    memory = system.get('memory', 'Unknown')
                    gpu_info = ', '.join(system.get('gpu', ['Unknown']))
                    entry['system_info'] = f"CPU: {cpu_info} | OS: {os_name} | Mem: {memory} | GPU: {gpu_info}"
                else:
                    entry['system_info'] = "N/A"

                # Extract experiment info (args)
                if 'args' in data['config']:
                    entry['experiment_info'] = ' '.join(data['config']['args'])
                else:
                    entry['experiment_info'] = "N/A"

                # Full JSON details
                entry['details'] = json.dumps(data, indent=2)

                df_tmp = pd.DataFrame(entry, index=[0])
                # rename columns that start with 'config.'
                df_tmp = df_tmp.rename(columns={c: c.split('config.')[-1] for c in df_tmp.columns})
                # replace . with _ in column names
                df_tmp.columns = [c.replace('.', '_') for c in df_tmp.columns]

                # For throughput benchmarks (ConstantVUs), rate is often None.
                # Use request_rate (achieved QPS) if rate is None.
                if 'rate' in df_tmp.columns and (df_tmp['rate'].isnull().all() or df_tmp['rate'].iloc[0] is None):
                     if 'request_rate' in df_tmp.columns:
                         df_tmp['rate'] = df_tmp['request_rate']
                elif 'rate' not in df_tmp.columns and 'request_rate' in df_tmp.columns:
                     df_tmp['rate'] = df_tmp['request_rate']

                df = pd.concat([df, df_tmp])
    return df


def build_results_df(results_dir) -> pd.DataFrame:
    df = pd.DataFrame()
    # list directories
    directories = [f'{results_dir}/{d}' for d in os.listdir(results_dir) if os.path.isdir(f'{results_dir}/{d}')] + [results_dir]
    for directory in directories:
        # list json files in results directory
        data_files = {}
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                data_files[filename.split('.')[-2]] = f'{directory}/{filename}'
        df = pd.concat([df, build_df(directory.split('/')[-1], data_files)])
    return df


def build_results(results_dir, results_file, device):
    df = build_results_df(results_dir)
    if 'device' not in df.columns:
        df['device'] = df['model'].apply(lambda x: device)
    df['error_rate'] = df['failed_requests'] / (df['failed_requests'] + df['successful_requests']) * 100.0
    df['prompt_tokens'] = df['total_tokens_sent'] / df['successful_requests']
    df['decoded_tokens'] = df['total_tokens'] / df['successful_requests']
    df.to_parquet(results_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results', type=str, required=True,
                        help='Path to the source directory containing the results')
    parser.add_argument('--results-file', type=str, required=True,
                        help='Path to the results file to write to. Can be a S3 path')
    parser.add_argument('--device', type=str, required=True, help='GPU name used for benchmarking')
    args = parser.parse_args()
    build_results(args.results_dir, args.results_file, args.device)
