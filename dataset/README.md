# GSM8K-AI-SubQ data
This folder contains the proposed dataset. See 
[data_generation_and_evaluation](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/data_generation_and_evaluation)
for description how the files are obtained.

[qg_train_data_raw.jsonl](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/dataset/qg_train_data_raw.jsonl)
and
[qg_test_data_raw.jsonl](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/dataset/qg_test_data_raw.jsonl)
are raw dataset with all sub-questions, corresponding answers and feedback (in case of train) produced by
ChatGPT.

[qg_train_dataset.jsonl](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/dataset/qg_train_dataset.jsonl) 
and 
[qg_test_dataset.jsonl](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/dataset/qg_test_dataset.jsonl)
is a form of the dataset that is used for training baselines in our work. It contains only input problems, sub-questions and rewards. 

See [baselines](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/baselines)
for examples of the dataset usage.

[test_chat.jsonl](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/dataset/test_chat.jsonl)
is a result of ChatGPT answering its own sub-questions on GSM8K test dataset.
