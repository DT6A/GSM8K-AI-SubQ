import argparse
import asyncio
import json
import os
import random
from time import time

import openai
import tqdm


openai.api_key = os.getenv("OPENAI_API_KEY")

message_qa = """You are given the mathematical problem marked with "Problem" and a sequence of sub-questions for solving it. Sub-question number N is marked as "QN:". Based on the problem for each sub-questions decide whether this question is helpful for solving the given problem. An essential property of a good questioning strategy is to ask questions that are directed towards the most critical domain specific content.  Asking the right sequence of relevant questions that can assist in reaching the final goal  is an important part of good questioning. If question repeats any of the previous it is not useful. The question for which answer is given in the problem or can't be answered at all is also not useful. So redundant questions are not good. 

For each question output me "QN: <Yes/No>" and only it where N is the number of the question, e.g. "Q1: <Yes/No> Q2: <Yes/No> for the first two questions. Do not try to solve the problem anyhow as I'm only interested in the quality of the sub-questions. Strictly follow the output format. Provide answers only for the last given problem.

Problem: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

Q1: How many eggs does Janet sell?
Q2: Is duck an animal? 
Q3: How many eggs does each duck lay? 
Q4: How much does Janet make at the farmers' market?

Q1: Yes
Q2: No
Q3: No
Q4: Yes

Problem: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?

Q1: How many bolts of white fiber does it take? 
Q2: How bolts of blue fiber does it take? 
Q3: How bolts of white fiber does it take? 
Q4: How many bolts in total does it take? 

Q1: Yes
Q2: No
Q3: No
Q4: Yes

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


async def async_main(args, load_name=f"generated_questions.jsonl", save_name=f"generated_feedback.jsonl"):
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
            lin_problems += [prefix] * args.n_feedbacks
    # print(*lin_problems)
    max_concurrent_tasks = args.parallel_queries
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    # timeout = 30  # seconds

    tasks = [async_query(p, semaphore) for p in lin_problems]
    results = await asyncio.gather(*tasks)
    responses = [r.choices[0].message.content if r is not None else None for r in results]

    mapped_responses = {}
    idx = 0
    for p, example in enumerate(data):
        mapped_responses[p] = {
            "problem": example["problem"],
            "questions": example["questions"],
            "feedback": [],
        }
        for i in range(len(example["questions"])):
            feedbacks = []
            for j in range(args.n_feedbacks):
                feedbacks.append(responses[idx])
                idx += 1
            mapped_responses[p]["feedback"].append(feedbacks)

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
    parser.add_argument("--n_feedbacks", type=int, default=3)
    parser.add_argument("--parallel_queries", type=int, default=3)
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
            save_name=f"generated_feedback_{shard_n}.jsonl",
            load_name=f"generated_questions_{shard_n}.jsonl",
        ))
        end = time()
        print(f"Took {end - start} seconds fo finish shard {shard_n}")
        shard_n += 1
