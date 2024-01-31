import json

import torch as th


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def create_qg_data(examples):
    clean_examples = []
    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        clean_examples.append(ex)

    return clean_examples


def question_generation(data_dir, split, train_frac=0.9, filtered_bc=False):
    path = f"{data_dir}/qg_{split}_dataset.jsonl"

    with open(path, "r") as file:
        line = file.readline()
        all_data = json.loads(line)
    examples = []
    last_idxs = []
    for data in all_data:
        if split == "train" and filtered_bc and data["rewards"]["answer_correctness"] == 0.0:
            continue
        temp = data
        all_previous_qs = ""
        for q in data["answers"]:
            all_previous_qs += f" <QUE> {q} </QUE>"
        temp["answer"] = all_previous_qs
        del temp["answers"]
        examples.append(temp)
        last_idxs.append(len(examples))
    split_idx = last_idxs[min(int(train_frac * (len(last_idxs))), len(last_idxs) - 1)]

    train_samples = create_qg_data(examples[:split_idx])
    valid_samples = create_qg_data(examples[split_idx:])

    print(f"{len(train_samples)} {split} examples")
    return train_samples, valid_samples


def generated_subques(path):
    all_data = read_jsonl(path)
    examples = all_data[0]

    print(f"{len(examples)} generated examples")
    return examples


class GSMDataset(th.utils.data.Dataset):
    def __init__(self, tokenizer, examples, loss_on_prefix=True, world_size=1, rank=0, additional_padding=0):
        self.world_size = world_size
        self.rank = rank
        self.examples = examples
        self.qns = [ex["question"] for ex in self.examples]
        self.ans = [ex["answer"] for ex in self.examples]
        qns = tokenizer(self.qns, padding=False)
        ans = tokenizer(self.ans, padding=False)
        fans = []
        fqns = []
        filtered_examples = []
        for i in range(len(self.examples)):
            if len(qns["input_ids"][i]) + len(ans["input_ids"][i]) >= tokenizer.model_max_length:
                continue
            fans.append(self.ans[i])
            fqns.append(self.qns[i])
            filtered_examples.append(self.examples[i])
        self.examples = filtered_examples

        self.qns = tokenizer(fqns, padding=False)
        self.ans = tokenizer(fans, padding=False)

        self.loss_on_prefix = loss_on_prefix
        self.max_len = max(
            [
                len(self.qns["input_ids"][i]) + len(self.ans["input_ids"][i])
                for i in range(len(fqns))
            ]
        ) + additional_padding
        print(f"Max tokens: {self.max_len}")

    def __len__(self):
        return len(self.examples) // self.world_size

    def __getitem__(self, idx):
        idx = idx * self.world_size + self.rank

        qn_tokens = self.qns["input_ids"][idx]
        ans_tokens = self.ans["input_ids"][idx]
        prefix_length = len(qn_tokens)
        sample_length = prefix_length + len(ans_tokens)
        pad_tokens = [0] * (self.max_len - len(qn_tokens) - len(ans_tokens))
        tokens = qn_tokens + ans_tokens + pad_tokens
        mask = (
                ([int(self.loss_on_prefix)] * len(qn_tokens))
                + ([1] * len(ans_tokens))
                + ([0] * len(pad_tokens))
        )
        prefix_mask = (
                ([1] * len(qn_tokens))
                + ([0] * len(ans_tokens))
                + ([0] * len(pad_tokens))
        )
        tokens = th.tensor(tokens)
        mask = th.tensor(mask)
        prefix_mask = th.tensor(prefix_mask)
        prefix_length = th.tensor(prefix_length)
        sample_length = th.tensor(sample_length)
        return dict(
            input_ids=tokens,
            attention_mask=mask,
            prefix_mask=prefix_mask,
            prefix_length=prefix_length,
            sample_length=sample_length,
        )


class GSMIQLDatasetQG(th.utils.data.Dataset):
    def __init__(self, tokenizer, examples, loss_on_prefix=True, world_size=1, rank=0, train_frac=1.0,
                 tokens_limit=None, sparse=False, seq_reward_mult=1.0, full_sparse=False):
        self.world_size = world_size
        self.rank = rank
        self.examples = examples[:int(len(examples) * train_frac)]
        self.qns = [ex["question"] for ex in self.examples]

        def tokenize_answers(answers, rewards):
            for i in range(len(answers)):
                answers[i] = f" <QUE> {answers[i]} </QUE>"
            tokens = tokenizer(answers, padding=False)
            answerwise_rewards = []
            seq_reward = 0.0
            for i, t in enumerate(tokens["input_ids"]):
                rs = [0] * len(t)
                for r in rewards:
                    if type(rewards[r]) is list:
                        if not sparse and not full_sparse:
                            rs[-1] += rewards[r][i]
                        if full_sparse and not (rewards[r][i] < 1.0):
                            seq_reward += rewards[r][i]

                answerwise_rewards.append(rs)
            for r in rewards:
                if type(rewards[r]) is not list:
                    seq_reward += rewards[r]
            answerwise_rewards[-1][-1] += seq_reward * seq_reward_mult
            tokens_res = []
            for t in tokens['input_ids']:
                tokens_res += t
            rewards_res = []
            for r in answerwise_rewards:
                rewards_res += r
            return tokens_res, rewards_res

        self.ans = []
        self.rs = []
        for ex in self.examples:
            rk = "rewards" if "rewards" in ex else "reward"
            a, r = tokenize_answers(ex["answers"], ex[rk])
            self.ans.append(a)
            self.rs.append(r)
        self.qns = tokenizer(self.qns, padding=False)
        self.loss_on_prefix = loss_on_prefix

        fans = []
        fqns_ids = []
        frs = []
        filtered_examples = []
        for i in range(len(self.examples)):
            if len(self.qns["input_ids"][i]) + len(self.ans[i]) >= tokenizer.model_max_length:
                continue
            if tokens_limit is not None and len(self.qns["input_ids"][i]) + len(self.ans[i]) >= tokens_limit:
                continue
            fans.append(self.ans[i])
            fqns_ids.append(i)
            frs.append(self.rs[i])
            filtered_examples.append(self.examples[i])
        self.examples = filtered_examples

        for k in self.qns:
            self.qns[k] = [self.qns[k][idx] for idx in fqns_ids]
        self.ans = fans
        self.rs = frs
        assert len(self.ans) == len(self.rs)
        for a, r in zip(self.ans, self.rs):
            assert len(a) == len(r)
        assert len(self.ans) == len(self.qns["input_ids"])

        self.max_len = max(
            [
                len(self.qns["input_ids"][i]) + len(self.ans[i])
                for i in range(len(self.examples))
            ]
        )
        print(f"Max tokens: {self.max_len}")

    def __len__(self):
        return len(self.examples) // self.world_size

    def __getitem__(self, idx):
        idx = idx * self.world_size + self.rank

        qn_tokens = self.qns["input_ids"][idx]
        ans_tokens = self.ans[idx]
        prefix_length = len(qn_tokens)
        sample_length = prefix_length + len(ans_tokens)
        pad_tokens = [0] * (self.max_len - len(qn_tokens) - len(ans_tokens))
        tokens = qn_tokens + ans_tokens + pad_tokens
        mask = (
                ([int(self.loss_on_prefix)] * len(qn_tokens))
                + ([1] * len(ans_tokens))
                + ([0] * len(pad_tokens))
        )
        prefix_mask = (
                ([1] * len(qn_tokens))
                + ([0] * len(ans_tokens))
                + ([0] * len(pad_tokens))
        )
        dones = (
                ([0] * len(qn_tokens))
                + ([0] * (len(ans_tokens) - 1))
                + ([1] * (len(pad_tokens) + 1))
        )
        # print(self.rs[idx])
        rewards = (
                ([0] * len(qn_tokens))
                + self.rs[idx]
                + ([0] * (len(pad_tokens)))
        )
        tokens = th.tensor(tokens)
        mask = th.tensor(mask)
        prefix_mask = th.tensor(prefix_mask)
        prefix_length = th.tensor(prefix_length)
        sample_length = th.tensor(sample_length)
        dones = th.tensor(dones)
        rewards = th.tensor(rewards)
        return dict(
            input_ids=tokens,
            attention_mask=mask,
            prefix_mask=prefix_mask,
            prefix_length=prefix_length,
            sample_length=sample_length,
            rewards=rewards,
            dones=dones,
        )
