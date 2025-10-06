import json

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

CACHE_DIR = '/home/adewinmb/orcd/scratch'

def generate_response(model_name: str, prompts: list[str]) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    
    results = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors='pt').to(model.device)
        
        out = model.generate(
            **ids,
            max_new_tokens=4,
            do_sample=False
        )
        
        # Extract integer from response
        response = "".join(char for char in tokenizer.decode(out[0]) if char.isdigit())
        results.append(int(response) if response != '' else None)

    return results

def benchmark(models: list[str], 
              datasets: list[str],
              n_samples: int) -> pd.DataFrame:
    """Benchmark models

    Args:
        models (list[str]): List of huggingface model names.
        datasets (list[str]): List of paths to datasets.
        n_samples (int): Number of samples to take from each dataset
        
    Returns:
        pd.DataFrame: DataFrame with columns:
            model (str): HF model name.
            dataset (str): Dataset path.
            sample_idx (int): The index of the sample in the dataset.
            output (int): The model's guess. Can be NaN.
            label (int): The true value.
    """
    benchmark_results = {
        'dataset': [],
        'sample_idx': [],
        'label': []
    }
    benchmark_results.update({model: [] for model in models})
        
    for model_idx, model in enumerate(models):
        print(f"Starting {model} eval")
        
        prompts = []
        for dataset in datasets:
            data = None
            with open(dataset, 'r') as f:
                data = json.load(f)
            
            # Generate prompts
            for sample in data[:n_samples]:
                formatted_list = ""
                for i in sample['list']:
                    formatted_list += i + ' '
                formatted_list = '[' + formatted_list.strip() + ']'
                
                p = f"""Count the number of words in the following list that match 
                the given type, and put the numerical answer in parentheses.\n\n
                Type: {sample['category']}\nList: {formatted_list}\nAnswer: ("""
                
                prompts.append(p)
            
        # Add entry for each response
        # Note: Can optimize to not iterate through results again
        for idx, result in enumerate(generate_response(model, prompts)):
            benchmark_results[model].append(result)
            if model_idx == 0:
                benchmark_results['dataset'].append( # Extract dataset name
                    datasets[idx//n_samples].split('/')[-1].split('.')[0])
                benchmark_results['sample_idx'].append(idx)
                benchmark_results['label'].append(data[idx]['count'])

        print(f"Finished eval for {model}\n")
        
        # Checkpoint
        model_safe = model.replace('/', '-')  # Replace / with - for valid filename
        pd.DataFrame({k: pd.Series(v) for k, v in benchmark_results.items()}).to_csv(
            f'benchmark-{model_safe}.csv',
            index=False
        )

    return pd.DataFrame(benchmark_results)

if __name__=='__main__':
    models_all = [
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.3-70B-Instruct",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        "google/gemma-3-4b-it",
        "google/gemma-3-1b-it",
        "google/gemma-3-4b-pt",
        "openai/gpt-oss-20b",
        # "openai/gpt-oss-120b", # TODO: need to debug this quantization
        "deepseek-ai/deepseek-llm-7b-chat"
    ]
    
    # <= 4B
    models_small = [
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-3B",
        "Qwen/Qwen3-4B",
        "google/gemma-3-4b-it",
        "google/gemma-3-1b-it",
        "google/gemma-3-4b-pt"
    ]
    
    models_smoketest = [
        "meta-llama/Llama-3.2-3B",
        "google/gemma-3-4b-it"
    ]
    
    datasets_all = [
        '../data_gen/data_trivial.json',
        '../data_gen/data_easy.json',
        '../data_gen/data_medium.json',
        '../data_gen/data_hard.json'
    ]
    
    datasets_smoketest = [
        '../data_gen/data_trivial.json',
        '../data_gen/data_hard.json',
    ]
    
    result_df = benchmark(models_all, datasets_all, 5000)
    result_df.to_csv('benchmark.csv', index=False)
    
