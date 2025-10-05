import json
import torch
import pandas
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_response(model_name: str, prompts: list[str]):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    results = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors='pt').to(model.device)
        
        out = model.generate(
            **ids,
            max_new_tokens=4,
            do_sample=False
        )
        
        # Extract integer from response
        num = int("".join(char for char in tokenizer.decode(out[0]) if char.isdigit()))
        results.append(num)
        print(f"Response: {num}")

    return results

if __name__=='__main__':
    # models = ["meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen3-4B", "google/gemma-3-4b-it", "deepseek-ai/deepseek-llm-7b-chat"]
    models = ["openai/gpt-oss-20b"]

    data = None
    with open('data_gen/data.json', 'r') as f:
        data = json.load(f)

    benchmark_results = {
        "label": [],
        "meta-llama/Llama-3.2-3B-Instruct": [],
        "Qwen/Qwen3-4B": [],
        "google/gemma-3-4b-it": [],
        "deepseek-ai/deepseek-llm-7b-chat": []
    }
    
    prompts = []
    
    # Create prompts
    for sample in data[:5]:
        formatted_list = ""
        for i in sample['list']:
            formatted_list += i + ' '
        formatted_list = '[' + formatted_list.strip() + ']'
        
        p = f"""Count the number of words in the following list that match 
        the given type, and put the numerical answer in parentheses.\n\n
        Type: {sample['category']}\nList: {formatted_list}\nAnswer: ("""
        
        prompts.append(p)
    
    for model in models:
        print(f"Starting {model} eval")
        benchmark_results[model] = generate_response(model, prompts)
        print(f"{model} results: {benchmark_results[model]}")
    
