import argparse
import asyncio
import json
import os
import random
from time import time

import openai
import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY")

message_qa = """You are given the mathematical problems marked with "Problem" and a sequence of questions which should help in solving it. Question number N is marked as "QN:". Based on the problem and sequence of questions answer each of the questions with answer "AN:" and give the answer to the whole problem using "Final answer:" using only the resulting number without adding any additional comments after it.

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
            except openai.error.InvalidRequestError as e:
                print("Skipping invalid query", e)
                return None
            except Exception as e:
                print("Some problem occurred, retrying in 5 sec. Stop the program if it still appears!")
                print(e)
                await asyncio.sleep(5)


async def async_main(args, load_name=f"generated_questions.jsonl", save_name=f"generated_answers.jsonl"):
    problems_dict = {}
    task2problem = {}
    load_path = os.path.join(args.load_dir, load_name)
    data = []
    with open(load_path, "r") as json_file:
        for line in json_file:
            data_dict = json.loads(line)
            data.append(data_dict)
    subset_size = min(args.n_proc_samples, len(data))
    if args.n_proc_samples == -1:
        subset_size = len(data)
    data = data[:subset_size]
    lin_problems = []
    for p, example in enumerate(data):
        problems_dict[p] = example
        for questions in example['questions']:
            prefix = message_qa + f"Problem: {example['problem']}\n\n{questions}\n\n"
            lin_problems.append(prefix)

    max_concurrent_tasks = 10
    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    tasks = [async_query(p, semaphore) for p in lin_problems]
    results = await asyncio.gather(*tasks)
    responses = [r.choices[0].message.content if r is not None else None for r in results]

    mapped_responses = {}
    idx = 0
    for p, example in enumerate(data):
        mapped_responses[p] = {
            "problem": example["problem"],
            "questions": example["questions"],
            "answers": [],
            "answer": example["answer"],
        }
        for _ in range(len(example["questions"])):
            mapped_responses[p]["answers"].append(responses[idx])
            idx += 1

    path = os.path.join(args.save_dir, save_name)
    print("Saving data to:", path)
    with open(path, "w") as json_file:
        for p in range(len(data)):
            json_file.write(json.dumps(mapped_responses[p]) + '\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_shard", type=int, default=0)
    parser.add_argument("--end_shard", type=int, default=74)
    parser.add_argument("--n_proc_samples", type=int, default=-1)
    parser.add_argument("--parallel_queries", type=int, default=10)
    parser.add_argument("--load_dir", type=str, default="chat_data")
    parser.add_argument("--save_dir", type=str, default="chat_data")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(0)
    shard_n = args.start_shard
    for i in range(args.start_shard, args.end_shard + 1):
        start = time()
        asyncio.run(async_main(
            args,
            save_name=f"generated_answers_{shard_n}.jsonl",
            load_name=f"generated_questions_{shard_n}.jsonl",
        ))
        end = time()
        print(f"Took {end - start} seconds fo finish shard {shard_n}")
        shard_n += 1
