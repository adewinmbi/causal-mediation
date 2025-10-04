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
