import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score

from scipy.stats import pearsonr, pointbiserialr

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
    if s is None:
        return None
    s = s.lower()
    pattern = "final answer:"
    idx = s.find(pattern) + len(pattern)
    return extract_first_number(s[idx:])


def extract_feedback_confidence(feedbacks):
    n_feedbacks = len(feedbacks)
    n_questions = max(map(lambda f: len(f.split("\n")), feedbacks))
    scores = np.array([0 for _ in range(n_questions)])
    for f in feedbacks:
        for i, fr in enumerate(map(lambda x: "yes" in x, f.lower().split("\n"))):
            if fr:
                scores[i] += 1
    return scores / n_feedbacks


def process_data(path, return_split=False, save=False):
    data = []
    with open(path, "r") as json_file:
        for line in json_file:
            data_dict = json.loads(line)
            data.append(data_dict)

    max_questions = max(map(lambda x: len(x['questions']), data))
    n_correct = 0
    n_all = 0
    correct_groups = {
        i: [] for i in range(max_questions + 1)
    }
    n_problems = len(data)
    n_questions = []
    questions_count_distr = []

    complexity_groupping = {}

    for sample in data:
        true_answer = extract_true_answer(sample["answer"])
        cur_corr = 0
        for i, ans in enumerate(sample["answers"]):
            questions = sample["questions"][i].split('\n')
            n_questions.append(len(questions))
            # if len(questions) > 20:
            #     continue
            questions_count_distr.append(len(questions))
            pred_answer = extract_predicted_answer(ans)
            n_all += 1
            if pred_answer == true_answer:
                cur_corr += 1
                n_correct += 1
        correct_groups[cur_corr].append(sample)

    complexity_groupping = {
        i: [] for i in range(max_questions + 1)
    }
    correct_prob = []
    for j in range(max_questions + 1):
        complexity_groupping[j] = list(map(lambda x: x["problem"], correct_groups[j]))
        correct_prob += [j] * len(complexity_groupping[j])
    correct_prob = np.array(correct_prob) / 3

    print("Overall accuracy:", n_correct / n_all)
    # print("Avg of avg:", np.mean(correct_prob), "+-", np.std(correct_prob))
    print("Avg number of questions:", np.mean(questions_count_distr), "+-", np.std(questions_count_distr))
    print("Median number of questions:", np.median(questions_count_distr))
    for k in correct_groups:
        print(f"Fraction of problems with {k} correct:", len(correct_groups[k]), len(correct_groups[k]) / n_problems)
    plt.hist(questions_count_distr, edgecolor='black', bins=[i for i in range(17)])
    plt.title("Number of sub-questions distribution in train set")
    plt.xlabel("Number of questions")
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig("imgs/questions_distr.pdf", dpi=300)
    plt.close()

    question_counts_0 = []
    for sample in correct_groups[max_questions]:
        questions = sample["questions"][i].split('\n')
        question_counts_0.append(len(questions))

    for k in correct_groups:
        if k == max_questions:
            continue
        question_counts = []
        for sample in correct_groups[k]:
            questions = sample["questions"][i].split('\n')
            question_counts.append(len(questions))
        plt.hist(question_counts_0, bins=[i for i in range(17)], edgecolor='black', alpha=0.5,
                 label=f"{max_questions} correct", density=True)
        plt.hist(question_counts, bins=[i for i in range(17)], edgecolor='black', alpha=0.5, label=f"{k} correct",
                 density=True)
        plt.title(f"Comparison of number of questions distributions {max_questions} vs. {k} correct answers")
        plt.xlabel("Number of questions")
        plt.ylabel("Density")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        if save:
            plt.savefig(f"imgs/questions_comp_{k}.pdf", dpi=300)
        plt.close()
    if return_split:
        return complexity_groupping


def process_feedback(path):
    data = []
    sample_idx = 0
    with open(path, "r") as json_file:
        for line in json_file:
            data_dict = json.loads(line)
            data.append(data_dict)
    max_questions = max(map(lambda x: len(x['questions']), data))
    n_correct = 0
    n_all = 0
    n_problems = len(data)
    questions_count_distr = []

    answer_incorrect = []
    negative_feedback = []
    correctness_confidence = []
    all_confidences = []
    avg_confs = []
    for idx, sample in enumerate(data):
        true_answer = extract_true_answer(sample["answer"])
        cur_corr = 0
        for i, ans in enumerate(sample["answers"]):
            questions = sample["questions"][i].split('\n')
            feedback = sample["feedback"][i]
            feedback_cat = '\n'.join(feedback)
            # if len(questions) > 20:
            #     continue
            pred_answer = extract_predicted_answer(ans)
            answer_incorrect.append(pred_answer != true_answer)
            cur_confidences = list(extract_feedback_confidence(feedback))
            avg_confs.append(np.mean(cur_confidences))
            all_confidences += cur_confidences
            negative_feedback.append("no" in feedback_cat.lower())

    print("Precision:", precision_score(answer_incorrect, negative_feedback))
    print("Recall:", recall_score(answer_incorrect, negative_feedback))
    print("ROC AUC:", roc_auc_score(answer_incorrect, 1 - np.array(avg_confs)))
    print("Fraction of problems with negative feedback:", np.array(negative_feedback).mean())
    cm = confusion_matrix(answer_incorrect, negative_feedback)
    # Plot confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                cmap="Blues")  # , xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Got negative feedback")
    plt.ylabel("Incorrect answer")
    plt.tight_layout()
    plt.savefig(f"imgs/feedback_cm.pdf", dpi=300)
    plt.close()

    plt.hist(all_confidences, edgecolor='black', )
    # plt.yscale("log")
    plt.title("All confidences distribution")
    plt.grid()
    plt.tight_layout()
    plt.xlabel("Usefulness score")
    plt.ylabel("Number of subquestions")
    plt.savefig(f"imgs/feedback_distr.pdf", dpi=300)
    plt.close()

    plt.hist(avg_confs, edgecolor='black', )
    plt.yscale("log")
    plt.title("Averaged confidences for sub-questions sets distribution")
    plt.grid()
    plt.tight_layout()
    plt.xlabel("Usefulness score avereged for each sub-questions set")
    plt.ylabel("Log number of sub-questions sets")
    plt.savefig(f"imgs/avg_feedback_distr.pdf", dpi=300)
    plt.close()

    corr, p_val = pearsonr(avg_confs, answer_incorrect)
    print(f"Pearson correlation coefficient is {corr} with p-value {p_val}")


def proc_eval(path, plots=False):
    data = []
    with open(path, "r") as json_file:
        for line in json_file:
            data_dict = json.loads(line)
            data.append(data_dict)
    max_questions = 1
    n_correct = 0
    n_all = 0
    correct_groups = {
        i: [] for i in range(max_questions + 1)
    }
    n_problems = len(data)
    questions_count_distr = []

    for sample in data:
        true_answer = sample["answer"]
        cur_corr = 0
        ans = sample["answers"]
        questions = sample["questions"].split('\n')
        # if len(questions) > 20:
        #     continue
        questions_count_distr.append(len(questions))
        pred_answer = extract_predicted_answer(ans)
        n_all += 1
        if pred_answer == true_answer:
            cur_corr += 1
            n_correct += 1
        correct_groups[cur_corr].append(sample)

    # print("Overall accuracy:", n_correct / n_all)
    # for k in correct_groups:
    #     print(f"Fraction of problems with {k} correct:", len(correct_groups[k]) / n_problems)
    if plots:
        plt.hist(questions_count_distr, edgecolor='black', bins=[i for i in range(17)])
        plt.title("Number of questions distribution")
        plt.xlabel("Number of questions")
        plt.show()

    question_counts_0 = []
    for sample in correct_groups[max_questions]:
        questions = sample["questions"].split('\n')
        question_counts_0.append(len(questions))

    if plots:
        for k in correct_groups:
            if k == max_questions:
                continue
            question_counts = []
            for sample in correct_groups[k]:
                questions = sample["questions"].split('\n')
                question_counts.append(len(questions))
            plt.hist(question_counts_0, bins=[i for i in range(17)], edgecolor='black', alpha=0.5,
                     label=f"{max_questions} correct", density=True)
            plt.hist(question_counts, bins=[i for i in range(17)], edgecolor='black', alpha=0.5, label=f"{k} correct",
                     density=True)
            plt.xlabel("Number of questions")
            plt.ylabel("Density")
            plt.legend()

            plt.show()
    return n_correct / n_all


def get_model_results(answers_dir):
    BC_files = [f"eval_finetune{ms}.jsonl" for ms in ["_dist", "", "_medium"]]
    FBC_files = [f"eval_filtered_bc{ms}.jsonl" for ms in ["_dist", "", "_medium"]]
    ILQL_full_files = [f"eval_iql{ms}.jsonl" for ms in ["_dist", "", "_medium"]]
    ILQL_sparse_files = [f"eval_iql{ms}_sparse.jsonl" for ms in ["_dist", "", "_medium"]]

    gpt_score = proc_eval(f"{answers_dir}/eval_chat.jsonl")

    scores = {
        "BC": [proc_eval(f"{answers_dir}/{fn}") for fn in BC_files],
        "Filtered BC": [proc_eval(f"{answers_dir}/{fn}") for fn in FBC_files],
        "ILQL sparse": [proc_eval(f"{answers_dir}/{fn}") for fn in ILQL_full_files],
        "ILQL full": [proc_eval(f"{answers_dir}/{fn}") for fn in ILQL_sparse_files],
    }

    print("Method\tDistillGPT\tGPT2 small\tGPT2 medium\tAvg")
    for k in scores:
        print(k, *scores[k], np.mean(scores[k]))
    print("ChatGPT", gpt_score)
    return scores, gpt_score


if __name__ == "__main__":
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.rcParams['figure.dpi'] = 300
    sns.set(style="ticks", font_scale=1.5)
    plt.rcParams.update({
        'font.serif': 'Times New Roman'
    })

    path_to_train_data = "../dataset/qg_train_data_raw.jsonl"
    os.makedirs("imgs", exist_ok=True)

    print("=" * 20, "Train data accuracy and questions distribution", "=" * 20)
    process_data(path_to_train_data, False, True)
    print('\n\n')

    print("=" * 20, "Feedback info", "=" * 20)
    process_feedback(path_to_train_data)
    print('\n\n')

    print("=" * 20, "Evaluation results", "=" * 20)
    print("=" * 5, "ChatGPT", "=" * 5)
    scores_chat, chat_chat = get_model_results("../eval_responses/chatgpt")
    print()
    print("=" * 5, "LLaMA 7B", "=" * 5)
    scores_l7b, chat_l7b = get_model_results("../eval_responses/llama7b")
    print()
    print("=" * 5, "LLaMA 13B", "=" * 5)
    scores_l13b, chat_l13b = get_model_results("../eval_responses/llama13b")
    print()
    print("=" * 5, "Mistral", "=" * 5)
    scores_m, chat_m = get_model_results("../eval_responses/mistral")
    print()

    print("=" * 5, "Scores avereged over models", "=" * 5)
    print("Method\tDistillGPT\tGPT2 small\tGPT2 medium\tAvg")
    for k in scores_chat:
        chat_s = scores_chat[k]
        llama7_s = scores_l7b[k]
        llama13_s = scores_l13b[k]
        mist_s = scores_m[k]
        means = np.mean([chat_s, llama7_s, llama13_s, mist_s], axis=0)
        print(k, *means, np.mean(means))
    print("ChatGPT", np.mean([chat_chat, chat_l7b, chat_l13b, chat_m]))
