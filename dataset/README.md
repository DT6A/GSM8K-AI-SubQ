# GSM8K-AI-SubQ data
This folder contains the proposed dataset. See 
[data_generation_and_evaluation](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/data_generation_and_evaluation)
for description how the files are obtained.

[qg_train_data_raw.jsonl](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/dataset/qg_train_data_raw.jsonl)
is a raw dataset with all sub-questions, corresponding answers and feedback produced by
ChatGPT.

[qg_train_dataset.jsonl](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/dataset/qg_train_dataset.jsonl)
is a form of the dataset that is used for training baselines in our work. It contains only input problems, sub-questions and rewards. 

[test_chat.jsonl](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/dataset/test_chat.jsonl)
is a result of ChatGPT answering it's own sub-questions on GSM8K test dataset.
