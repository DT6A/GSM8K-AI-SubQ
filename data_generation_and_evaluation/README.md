# Data generation and evaluation

## Train dataset generation

Data generation with ChatGPT processes data with shards in order to keep obtained data in case
of any technical issues so the collected data doesn't get lost. 

[chatgpt_qg.py](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/data_generation_and_evaluation/chatgpt_qg.py) 
generates sub-questions for the given GSM8K split.
```commandline 
OPENAI_API_KEY=<your_key> python3 chatgpt_gq.py \\ 
    --data_fold=[train/test] \\
    --start_sample=[id_of_sample_to_start_with] \\
    --end_sample=[id_of_the_last_sample_to_process] \\ 
    --n_repeats=[number_of_subquestion_sets_to_generate_for_each_problem] \\
    --n_samples=[number_of_samples_in_each_shard] \\
    --shard_n=[id_of_the_first_shard] \\
    --parallel_queries=[number_of_parallel_API_calls] \\
    --save_dir=[save_dir_name]
```
`data_fold`, `n_repeats`, `parallel_queries` and `save_dir` are the only parameters you probably need to change. 

Obtained sub-questions shards can be further processed with [chatgpt_qa.py](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/data_generation_and_evaluation/chatgpt_qa.py) 
script to get answers for each of the sub-questions and problem final answer.
```commandline 
OPENAI_API_KEY=<your_key> python3 chatgpt_ga.py \\ 
    --start_shard=[id_of_shard_to_start_with] \\
    --end_shard=[id_of_the_last_shard_to_process] \\ 
    --n_proc_samples=[number_of_samples_to_process] \\
    --parallel_queries=[number_of_parallel_API_calls] \\
    --load_dir=[path_to_subquestions_shards] \\
    --save_dir=[save_dir_name]
```

Sub-questions shards can be also processed with [chatgpt_feedback.py](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/data_generation_and_evaluation/chatgpt_feedback.py) 
script to obtain feedback on usefulness of each sub-question.
```commandline 
OPENAI_API_KEY=<your_key> python3 chatgpt_feedback.py \\ 
    --start_shard=[id_of_shard_to_start_with] \\
    --end_shard=[id_of_the_last_shard_to_process] \\ 
    --n_proc_samples=[number_of_samples_to_process] \\
    --parallel_queries=[number_of_parallel_API_calls] \\
    --load_dir=[path_to_subquestions_shards] \\
    --save_dir=[save_dir_name]
```

Generated answers and feedbacks shards can be merged into a single file using [merge_shards.py](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/data_generation_and_evaluation/merge_shards.py)

```commandline 
python3 merge_shards.py \\ 
    --shards_count=[number_of_shards_to_merge] \\
    --load_dir=[path_to_subquestions_shards] \\
    --save_path=[save_path] \\
    [--no_feedback (flag to ignore feedback)]
```

Result of [merge_shards.py](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/data_generation_and_evaluation/merge_shards.py) 
is a raw dataset and can be found here: [qg_train_data_raw.jsonl](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/dataset/qg_train_data_raw.jsonl).
This file can be converted into the format which is used for our baselines training using [convert_to_qg_dataset.py](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/data_generation_and_evaluation/convert_to_qg_dataset.py),
[qg_train_dataset.jsonl](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/dataset/qg_train_dataset.jsonl) 
is a result of this processing. It keeps only original problem, subquestions and numerical reward.

## Test data ChatGPT subq-uestions + answering
In order to obtain ChatGPT sub-questions and corresponding answers for the test set 
[chatgpt_qg.py](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/data_generation_and_evaluation/chatgpt_qg.py)
and
[chatgpt_qa.py](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/data_generation_and_evaluation/chatgpt_qa.py)
can be reused and 
[convert_chat_test.py](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/data_generation_and_evaluation/convert_chat_test.py)
used to convert shards into the proper evaluation format.

```commandline 
python3 convert_chat_test.py \\ 
    --n_shards=[number_of_shards_to_merge] \\
    --load_dir=[path_to_answers_shards] \\
    --save_dir=[save_directory] \\
```
It will save the file into the given directory with `test_chat.jsonl` name, see 
[test_chat.jsonl](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/dataset/test_chat.jsonl).

## Test set evaluation
All results of the scripts presented here are available for all baselines in
[eval_responses](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/eval_responses)

### ChatGPT
Use 
[chatgpt_run_test.py](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/data_generation_and_evaluation/chatgpt_run_test.py)
to evaluate results of our baselines with ChatGPT.
```commandline 
OPENAI_API_KEY=<your_key> python3 chatgpt_run_test.py \\ 
    --parallel_queries=[number_of_parallel_API_calls] \\
    --load_dir=[path_to_subquestions_shards] \\
    --load_name=[name_of_the_file_to_load] \\
    --save_dir=[save_dir_name]
    --save_name=[name_of_the_resulting_file]
```

### Other LMs
Use 
[llm_run_test.py](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/data_generation_and_evaluation/llm_run_test.py)
to evaluate results of our baselines with any HuggingFace LM checkpoint.
```commandline 
python3 llm_run_test.py \\ 
    --model_path=[path_to_huggingface_checkpoint_directory] \\
    --load_dir=[path_to_subquestions_shards] \\
    --load_name=[name_of_the_file_to_load] \\
    --save_dir=[save_dir_name]
    --save_name=[name_of_the_resulting_file]
```
