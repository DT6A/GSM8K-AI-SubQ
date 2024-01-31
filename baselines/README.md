# Baselines
This directory contains codebase for running experiments with provided baselines.

## Examples of training
The following commands will run training of all four presented approaches using GPT2 small as a backbone model.
Note that all approaches except BC must use BC checkpoint as model initialization to reproduce the reported scores.
Provided batch size and gradient accumulation parameters should fit on 32Gb V100 GPU. For GPT2 batch size 
and evaluation frequency  should be divided by 2 while gradient accumulation and number of training steps multiplied by 2.

### BC training
```commandline
python3 finetune_gpt_qg.py --model_path="gpt2" --tokenizer_path="gpt2" --save_dir="gpt_qg_checkpoints" --num_train_steps=40001 --save_interval=4000 --eval_interval=4000 --batch_size=8 --gradient_accumulation_steps=4 --run_name="GPT2 QG" --dataset_dir="../dataset"
```
### Filtered BC training
```commandline
python3 finetune_gpt_qg.py --model_path="gpt_qg_checkpoints/checkpoint-best" --save_dir="gpt_qg_fbc_checkpoints" --tokenizer_path=gpt2 --num_train_steps=30001 --save_interval=3000 --eval_interval=3000 --batch_size=8 --gradient_accumulation_steps=4 --run_name="GPT2 QG filtered BC" --filtered_bc --dataset_dir="../dataset"
```
### ILQL full training
```commandline
python3 ilql_gpt_qg.py --tokenizer_path=gpt2 --model_path="gpt_qg_checkpoints/checkpoint-best" --save_dir=gpt_ilql_qg_checkpoints --num_train_steps=200001 --save_interval=20000 --eval_interval=20000 --batch_size=4 --gradient_accumulation_steps=8 --run_name="ILQL QG" --cql_w=0.01 --iql_tau=0.9 --discount_factor=0.999 --dataset_dir="../dataset"
```
### ILQL sparse training
```commandline
python3 ilql_gpt_qg.py --tokenizer_path=gpt2 --model_path="gpt_qg_checkpoints/checkpoint-best" --save_dir=gpt_ilql_qg_sparse_checkpoints --num_train_steps=200001 --save_interval=20000 --eval_interval=20000 --batch_size=4 --gradient_accumulation_steps=8 --run_name="ILQL sparse QG" --cql_w=0.01 --iql_tau=0.9 --discount_factor=0.999 --sparse --dataset_dir="../dataset"
```

## Examples of predictions for the test set
Note, ILQL test predictions can be run together with training by adding `--eval` and setting up all the required arguments.
### BC
```commandline
python3 test_gpt_qg.py --tokenizer_path=gpt2 --model_path="gpt_qg_checkpoints/checkpoint-best" --batch_size=16 --run_name="GPT2 QG BC generating test" --save_dir="outputs" --save_name="finetune.jsonl"
```
### Filtered BC
```commandline
python3 test_gpt_qg.py --tokenizer_path=gpt2 --model_path="gpt_qg_fbc_checkpoints/checkpoint-best" --batch_size=16 --run_name="GPT2 QG Filtererd BC generating test" --save_dir="outputs" --save_name="filtered_bc.jsonl"
```
### ILQL full
```commandline
python3 ilql_gpt_qg.py --tokenizer_path=gpt2 --model_path="gpt_qg_checkpoints/checkpoint-best"  --save_dir="tmp"  --rl_path="gpt_ilql_qg_checkpoints/checkpoint-best" --num_train_steps=0  --save_interval=10 --eval_interval=10 --batch_size=8 --run_name="ILQL QG generating test" --dataset_dir="../dataset" --eval --eval_beta=1.0 --eval_dir="outputs" --save_name="iql.jsonl"
```
### ILQL sparse
```commandline
python3 ilql_gpt_qg.py --tokenizer_path=gpt2 --model_path="gpt_qg_checkpoints/checkpoint-best"  --save_dir="tmp"  --rl_path="gpt_ilql_qg_sparse_checkpoints/checkpoint-best" --num_train_steps=0  --save_interval=10 --eval_interval=10 --batch_size=8 --run_name="ILQL sparse QG generating test" --dataset_dir="../dataset" --eval --eval_beta=1.0 --eval_dir="outputs" --save_name="iql_sparse.jsonl"
```
Resulting output files can be further used with scripts from 
[data_generation_and_evaluation](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/data_generation_and_evaluation)
to get scores with different LMs.

## Commands arguments
### BC/Filtered BC train
```commandline 
python3 finetune_gpt_qg.py \\ 
    --model_path=[path_to_model_weights_or_huggingface_model_name] \\
    --tokenizer_path=[name_of_hugging_face_tokenizer] \\
    --save_dir=[directory_for_checkpoints] \\
    --dataset_dir=[path_to_dataset] \\
    --batch_size=[batch_size] \\
    --learning_rate=[learning_rate] \\
    --gradient_accumulation_steps=[number_of_gradient_accumulation_steps] \\
    --num_train_steps=[number_of_forward_passes] \\
    --save_interval=[checkpointing_interval] \\
    --eval_interval=[evaluation_interval] \\
    --run_name=[wandb_run_name] \\
    --seed=[random_seed] \\
    [--filtered_bc (flag to run Filtered BC)]
```

### BC/Filtered BC eval
```commandline 
python3 test_gpt_qg.py \\ 
    --model_path=[path_to_model_weights_or_huggingface_model_name] \\
    --tokenizer_path=[name_of_hugging_face_tokenizer] \\
    --save_dir=[directory_for_checkpoints] \\
    --dataset_dir=[path_to_dataset] \\
    --batch_size=[batch_size] \\
    --run_name=[wandb_run_name] \\
    --seed=[random_seed] \\
    --save_dir=[directory_to_save_results] \\
    --save_name=[file_name_for_results]
```

### ILQL
```commandline 
python3 ilql_gpt_qg.py \\ 
    --model_path=[path_to_model_weights_or_huggingface_model_name] \\
    --tokenizer_path=[name_of_hugging_face_tokenizer] \\
    --save_dir=[directory_for_checkpoints] \\
    --dataset_dir=[path_to_dataset] \\
    --batch_size=[batch_size] \\
    --learning_rate=[learning_rate] \\
    --gradient_accumulation_steps=[number_of_gradient_accumulation_steps] \\
    --num_train_steps=[number_of_forward_passes] \\
    --save_interval=[checkpointing_interval] \\
    --eval_interval=[evaluation_interval] \\
    --run_name=[wandb_run_name] \\
    --seed=[random_seed] \\
    --discount_factor=[discount_factor_value] \\
    --tau=[polyak_avereging_coefficient] \\
    --iql_tau=[iql_tau_hyperparameter] \\
    --cql_w=[weight_of_cql_term_in_loss] \\
    --v_w=[weight_of_v_function_in_loss] \\
    --q_w=[weight_of_q_function_in_loss] \\
    --eval_dir=[directory_to_save_results] \\
    --save_name=[file_name_for_results]
    --eval_beta=[ILQL_beta_hyperparameter_for_test_set_gereration] \\
    [--sparse (flag to run ILQL sparse)] \\
    [--eval (flag to run test set generation)] 
```
