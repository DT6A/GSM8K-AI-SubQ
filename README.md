# GSM8K-AI-SubQ
[![arXiv](https://img.shields.io/badge/arXiv-2402.01812-b31b1b.svg)](https://arxiv.org/abs/2402.01812)

This repository contains GSM8K-AI-SubQ dataset, scripts for its collection and scripts for baselines.

The dataset was created to conduct research in the direction of distillation of LLMs reasoning abilities,
particularly their ability of splitting problems into simpler sub-problems. 
We have employed ChatGPT for the generation of the dataset. It is based on GSM8K dataset and
includes examples of ChatGPT problems decomposition and its own feedback on generated sub-questions. 
Our data also includes ChatGPT's answers for sub-questions, but we didn't conduct any experiments for this part of reasoning. 
We hope that our dataset will help further advancements of offline RL algorithms in the area of reasoning.

For more details see our work ["Distilling LLMs' Decomposition Abilities into Compact Language Models"](https://arxiv.org/abs/2402.01812).

## Repository structure
Each of the directories contains README.md with relevant instructions and comments.
All the requirements can be installed with
```commandline
python3 -m pip install -r requirements.txt
```
* [baselines](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/baselines) contains the scripts of baseline algorithms: Behavioral Cloning (BC), Filtered BC and ILQL.
* [data_generation_and_evaluation](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/data_generation_and_evaluation) contains the scripts and data required for the generation of the dataset and scripts for evaluation of results.
* [dataset](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/dataset) contains the GSM8K-AI-SubQ dataset.
* [eval_responses](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/eval_responses) contains test set sub-questions generated with different baselines and answers of different language models to these sub-questions.
* [results_processing](https://github.com/DT6A/GSM8k-AI-SubQ/blob/main/results_processing) contains scripts for results processing.

## Evaluation results
### ChatGPT as sub-question answerer
| Algorithm   | DistillGPT | GPT-2 small | GPT-2 medium | Average |
|-------------|------------|-------------|--------------|---------|
| BC          | 0.476      | 0.508       | 0.538        | 0.507   | 
| Filtered BC | 0.493      | 0.527       | 0.576        | 0.532   | 
| ILQL-sparse | 0.471      | 0.518       | 0.541        | 0.510   | 
| ILQL-full   | 0.484      | 0.504       | 0.540        | 0.509   | 
| ChatGPT     | -          | -           | -            | 0.682   | 

### LLaMA 7B as sub-question answerer
| Algorithm   | DistillGPT | GPT-2 small | GPT-2 medium | Average |
|-------------|------------|-------------|--------------|---------|
| BC          | 0.118      | 0.154       | 0.164        | 0.145   | 
| Filtered BC | 0.125      | 0.159       | 0.162        | 0.149   | 
| ILQL-sparse | 0.125      | 0.138       | 0.162        | 0.142   | 
| ILQL-full   | 0.114      | 0.144       | 0.163        | 0.140   | 
| ChatGPT     | -          | -           | -            | 0.234   |

### LLaMA 13B as sub-question answerer
| Algorithm   | DistillGPT | GPT-2 small | GPT-2 medium | Average |
|-------------|------------|-------------|--------------|---------|
| BC          | 0.184      | 0.212       | 0.247        | 0.214   | 
| Filtered BC | 0.194      | 0.230       | 0.245        | 0.223   | 
| ILQL-sparse | 0.180      | 0.207       | 0.250        | 0.212   | 
| ILQL-full   | 0.182      | 0.210       | 0.252        | 0.215   | 
| ChatGPT     | -          | -           | -            | 0.353   |

### Mistral as sub-question answerer
| Algorithm   | DistillGPT | GPT-2 small | GPT-2 medium | Average |
|-------------|------------|-------------|--------------|---------|
| BC          | 0.240      | 0.264       | 0.290        | 0.265   | 
| Filtered BC | 0.228      | 0.256       | 0.293        | 0.259   | 
| ILQL-sparse | 0.219      | 0.261       | 0.288        | 0.256   | 
| ILQL-full   | 0.231      | 0.252       | 0.280        | 0.254   | 
| ChatGPT     | -          | -           | -            | 0.446   |

### Average among sub-question answerers
| Algorithm   | DistillGPT | GPT-2 small | GPT-2 medium | Average |
|-------------|------------|-------------|--------------|---------|
| BC          | 0.255      | 0.284       | 0.310        | 0.283   | 
| Filtered BC | 0.260      | 0.293       | 0.319        | 0.291   | 
| ILQL-sparse | 0.249      | 0.281       | 0.310        | 0.280   | 
| ILQL-full   | 0.253      | 0.277       | 0.309        | 0.280   |
| ChatGPT     | -          | -           | -            | 0.429   |

## Citing
If you use our work in your research, please use the following bibtex
```bibtex
@article{tarasov2024distilling,
  title={Distilling LLMs' Decomposition Abilities into Compact Language Models},
  author={Tarasov, Denis and Shridhar, Kumar},
  journal={arXiv preprint arXiv:2402.01812},
  year={2024}
}
```