import argparse
import asyncio
import json
import os
import random
from time import time

from vllm import LLM, SamplingParams


message_qa = """You are given the mathematical problems marked with "Problem" and a sequence of questions which should help in solving it. Question number N is marked as "QN:". Based on the problem and sequence of questions answer each of the questions with answer "AN:" and give the answer to the whole problem using "Final answer:" using only the resulting number without adding any additional comments after it. Do not come up with any additional problems and process only the last problem you are given.

Problem: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?

Q1: How many bolts of white fiber does it take? 
Q2: How many bolts in total does it take? 

A1: It takes 2/2=1 bolt of white fiber
A2: So the total amount of fabric is 2+1=3

Final answer: 3

Problem: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?

Q1: How much did the house cost?
Q2: How much did the repairs increase the value of the house? 
Q3: What is the new value of the house? 
Q4: How much profit did he make? 

A1: The cost of the house and repairs came out to 80,000+50,000=130,000
A2: He increased the value of the house by 80,000*1.5=120,000
A3: So the new value of the house is 120,000+80,000=200,000
A4: So he made a profit of 200,000-130,000=70,000

Final answer: 70000

"""


def query(prompt, llm, sampling_params):
    outputs = llm.generate([prompt], sampling_params)
    return outputs


def format_questions(questions):
    return "\n".join([f"Q{i + 1}: {questions[i]}" for i in range(len(questions))])


def main(args, load_name=f"generated_questions.jsonl", save_name=f"generated_answers.jsonl"):
    problems_dict = {}
    load_path = os.path.join(args.load_dir, load_name)
    with open(load_path, "r") as json_file:
        line = json_file.readline()
        data_dict = json.loads(line)
        data = data_dict
    lin_problems = []
    for p, example in enumerate(data):
        problems_dict[p] = example
        example['questions'] = format_questions(example['questions'])
        prefix = message_qa + f"Problem: {example['problem']}\n\n{example['questions']}\n\n"
        lin_problems.append(prefix)

    llm = LLM(args.model_path, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0.0, n=1, max_tokens=500)

    results = [query(p, llm, sampling_params) for p in lin_problems]
    responses = [r[0].outputs[0].text if r is not None else None for r in results]

    mapped_responses = {}
    idx = 0
    for p, example in enumerate(data):
        mapped_responses[p] = {
            "problem": example["problem"],
            "questions": example["questions"],
            "answers": [],
            "answer": example["answer"],
        }
        mapped_responses[p]["answers"] = responses[idx]
        idx += 1

    path = os.path.join(args.save_dir, save_name)
    print("Saving data to:", path)
    with open(path, "w") as json_file:
        for p in range(len(data)):
            json_file.write(json.dumps(mapped_responses[p]) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/mistral-7B")
    parser.add_argument("--load_dir", type=str, default="test_chat")
    parser.add_argument("--load_name", type=str, default="finetune.jsonl")
    parser.add_argument("--save_dir", type=str, default="eval_mistral")
    parser.add_argument("--save_name", type=str, default="eval_finetune.jsonl")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(0)
    start = time()
    main(
        args,
        save_name=args.save_name,
        load_name=args.load_name,
    )
    end = time()
    print(f"Took {end - start} seconds to finish")