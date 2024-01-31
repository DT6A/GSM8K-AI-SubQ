import argparse
import json
import os
import re

import numpy as np


def extract_first_number(s):
    number_pattern = r'[-+]?\d*\.\d+|\d+|\,\d+'
    match = re.search(number_pattern, s)
    if match:
        matched_number = match.group().replace(',', '')
        return float(matched_number)
    return None


def extract_true_answer(s):
    return extract_first_number(s)


def extract_predicted_answer(s):
    s = s.lower()
    pattern = "final answer:"
    idx = s.find(pattern) + len(pattern)
    return extract_first_number(s[idx:])


def extract_feedback_confidence(feedbacks, n_questions):
    n_feedbacks = len(feedbacks)
    # n_questions = max(map(lambda f: len(f.split("\n")), feedbacks))
    scores = np.array([0 for _ in range(n_questions)])
    for f in feedbacks:
        for i, fr in enumerate(map(lambda x: "yes" in x, f.lower().split("\n"))):
            if fr and i < n_questions:
                scores[i] += 1
    return scores / n_feedbacks


def extract_questions(questions):
    pattern = r"Q\d+: (.*)"
    extracted_text = [re.match(pattern, s).group(1) for s in questions.split("\n") if re.match(pattern, s)]
    return extracted_text


def extract_samples(sample):
    result = []
    problem = sample["problem"]
    true_answer = extract_true_answer(sample["answer"])
    if true_answer is None:
        print("\tFailed to extract the true answer")
        return result

    n_question_sets = len(sample["questions"])
    for i in range(n_question_sets):
        pred_answer = extract_predicted_answer(sample["answers"][i])
        questions = extract_questions(sample["questions"][i])
        usefulness = []
        if "feedback" in sample:
            usefulness = extract_feedback_confidence(sample["feedback"][i], len(questions))

        subsample = {
            "question": problem,
            "answers": questions,
            "rewards": {
                "answer_correctness": float(pred_answer == true_answer),
                "usefulness": list(usefulness)
            },
            "true_answer": true_answer,
        }
        result.append(subsample)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_name", type=str, default="chat_data/merged_data.jsonl")
    parser.add_argument("--save_name", type=str, default="chat_data/qg_dataset.jsonl")
    args = parser.parse_args()

    data = []
    sample_idx = 0
    with open(f"{args.load_name}", "r") as file:
        for i, line in enumerate(file):
            print("Processing sample", i + 1)
            data_dict = json.loads(line)
            data += extract_samples(data_dict)

    with open(args.save_name, "w") as json_file:
        json_file.write(json.dumps(data))
