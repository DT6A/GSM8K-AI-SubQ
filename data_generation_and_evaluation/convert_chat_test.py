import argparse
import os
import re
import json


def extract_first_number(s):
    number_pattern = r'[-+]?\d*\.\d+|\d+|\,\d+'
    match = re.search(number_pattern, s)
    if match:
        matched_number = match.group().replace(',', '')
        return float(matched_number)
    return None


def extract_true_answer(s):
    return extract_first_number(s)


def convert_test_data(path, shards_count, save_path):
    data = []
    for i in range(shards_count):
        with open(f"{path}/generated_answers_{i}.jsonl", "r") as json_file:
            for line in json_file:
                data_dict = json.loads(line)
                data_dict["answer"] = extract_true_answer(data_dict["answer"])
                data.append(data_dict)
    with open(f"{save_path}/test_chat.jsonl", "w") as json_file:
        json_file.write(json.dumps(data))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", type=str, default="chat_data_test")
    parser.add_argument("--save_dir", type=str, default="test_chat")
    parser.add_argument("--n_shards", type=int, default=14)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    convert_test_data(args.load_dir, args.n_shards, args.save_dir)
