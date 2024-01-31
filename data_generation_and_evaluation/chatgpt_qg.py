import argparse
import asyncio
import json
import os
import random
import re
from time import time

import openai


openai.api_key = os.getenv("OPENAI_API_KEY")

message_qg = """You are given mathematical problems marked with "Problem". Your task is to split it into smaller sub-problems and formulate  them as sub-questions which will be answered by someone else who's objective is to solve the original problem. Questions must not contain the answers for the previous questions in them. Do not ask questions where the answer is already given in the problem. For each problem come up with the sequence of sub-questions and output each of them on separate line which starts with letter Q followed by the number of question. Do not output anything else.

Problem: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?

Q1: How many bolts of white fiber does it take? 
Q2: How many bolts in total does it take? 

Problem: Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?

Q1: How much did the house cost?
Q2: How much did the repairs increase the value of the house? 
Q3: What is the new value of the house? 
Q4: How much profit did he make? 

"""


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def remove_calc(examples):
    processed = []
    for e in examples:
        proc_answers = list(
            map(
                lambda p: re.sub(r"<<.*?>>", "", p), e["answers"]
            )
        )
        tmp = dict(e)
        tmp["answers_text"] = proc_answers
        processed.append(tmp)

    return processed


def assist_subques_no_merge_only_text(split):
    path = os.path.join("data/", f"{split}_socratic.jsonl")
    all_data = read_jsonl(path)

    examples = []
    for data in all_data:
        problem = data["question"]
        qna_pairs = list(
            map(
                lambda p: p.split(" ** "), data["answer"].split("\n")
            )
        )
        examples.append({
            "problem": problem,
            "questions": [q[0] for q in qna_pairs[:-1]],
            "answers": [q[1] for q in qna_pairs[:-1]],
            "answer": qna_pairs[-1][0],
        })

    final_examples = remove_calc(examples)
    print(f"{len(final_examples)} {split} examples")
    return final_examples


async def async_query(prompt, semaphore):
    async with semaphore:
        while True:
            try:
                chat_completion_resp = await openai.ChatCompletion.acreate(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
                return chat_completion_resp
            except Exception as e:
                print("Some problem occurred, retrying in 5 sec. Stop the program if it still appears!")
                await asyncio.sleep(5)


async def async_main(args, save_name=f"generated_questions.jsonl"):
    problems_dict = {}
    data = assist_subques_no_merge_only_text(args.data_fold)[args.start_sample:]
    subset_size = min(args.n_samples, len(data))
    if args.n_samples == -1:
        subset_size = len(data)
    data = data[:subset_size]
    lin_problems = []
    for p, example in enumerate(data):
        problems_dict[p] = example
        prefix = message_qg + f"Problem: {example['problem']}\n\n"
        lin_problems += [prefix] * args.n_repeats

    max_concurrent_tasks = 10
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    tasks = [async_query(p, semaphore) for p in lin_problems]
    results = await asyncio.gather(*tasks)
    responses = [r.choices[0].message.content for r in results]

    mapped_responses = {}
    for p, example in enumerate(data):
        mapped_responses[p] = {
            "problem": example["problem"],
            "questions": [],
            "answer": example["answer"],
        }
        for i in range(args.n_repeats):
            mapped_responses[p]["questions"].append(responses[p * args.n_repeats + i])

    path = os.path.join(args.save_dir, save_name)
    print("Saving data to:", path)
    with open(path, "w") as json_file:
        for p in range(len(data)):
            json_file.write(json.dumps(mapped_responses[p]) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_fold", type=str, default="train")
    parser.add_argument("--start_sample", type=int, default=0)
    parser.add_argument("--end_sample", type=int, default=7473)
    parser.add_argument("--n_repeats", type=int, default=3)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--shard_n", type=int, default=0)
    parser.add_argument("--parallel_queries", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="chat_data")
    args = parser.parse_args()

    shard_n = args.shard_n
    for i in range(args.start_sample, args.end_sample, args.n_samples):
        os.makedirs(args.save_dir, exist_ok=True)
        random.seed(0)
        start = time()
        args.start_sample = i
        asyncio.run(async_main(args, save_name=f"generated_questions_{shard_n}.jsonl"))
        shard_n += 1
        end = time()
        print(f"Took {end - start} seconds fo finish shard {i}-{i + args.n_samples}")
