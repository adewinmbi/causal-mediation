# causal-mediation
In an LLM trained to count words belonging to a category, can we find a layer that keeps a running count?

## Task Statement (by David Bau)

Given the following type of prompt, a sufficiently large language model will be able to answer with the correct number. 

```
Count the number of words in the following  list that match the given type, and put the numerical answer in parentheses.

Type: fruit
List: [dog apple cherry bus cat grape bowl]
Answer: (
```

Your task:
1. create a dataset of several thousand examples like this.
2. benchmark some open-weight LMs on solving this task zero-shot (without reasoning tokens)
3. for a single model, create a causal mediation analysis experiment (patching from one run to another) to answer: "is there a hidden state layer that contains a representation of the running count of matching words, while processing the list of words?"

## Benchmark

Performace measured by model and by dataset difficulty using 5000 samples per difficulty level, or 20,000 samples per model. 

| Model                                |    MAE |   Error STD |
|:-------------------------------------|-------:|------------:|
| meta-llama/Llama-3.2-3B-Instruct     |   1.78 |        3.32 |
| meta-llama/Llama-3.2-3B              |   1.65 |        1.3  |
| meta-llama/Llama-3.3-70B-Instruct    | nan    |      nan    |
| Qwen/Qwen3-4B                        | nan    |      nan    |
| Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 | nan    |      nan    |
| google/gemma-3-4b-it                 | nan    |      nan    |
| google/gemma-3-1b-it                 | nan    |      nan    |
| google/gemma-3-4b-pt                 | nan    |      nan    |
| openai/gpt-oss-20b                   | nan    |      nan    |
| deepseek-ai/deepseek-llm-7b-chat     | nan    |      nan    |

Samples are categorized as either trivial, easy, medium, or hard. Harder samples have a higher frequency of potentially multisemantic words, for example, "orange" can be categorized as a fruit OR a color. This distinction was made to eventually uncover the causality behind whether a sequence was miscounted due to a misunderstanding of the word, or due to a lack of counting circuitry. I got the idea after seeing Claude Sonnet 4.5 fail to count what I thought was a simple sequence. 

| Dataset      |   MAE |   Error STD |
|:-------------|------:|------------:|
| data_trivial |  1.46 |        1.52 |
| data_easy    |  1.45 |        1.53 |
| data_medium  |  1.94 |        2.8  |
| data_hard    |  2    |        2.8  |
