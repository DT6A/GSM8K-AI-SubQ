import nltk
import numpy as np
import torch

import torch.nn.functional as F


def get_devices():
    return [
        torch.device(f"cuda:{i}")
        for i in range(torch.cuda.device_count())
    ]


class DatasetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (
            torch.LongTensor(self.dataset[idx]["input_ids"]),  # inputs
            torch.LongTensor(self.dataset[idx]["attention_mask"]),
            torch.LongTensor(self.dataset[idx]["input_ids"]),  # labels, shifted for HF GPT2 inside model's code
            torch.LongTensor(self.dataset[idx]["prefix_mask"]),
            torch.LongTensor(self.dataset[idx]["prefix_length"]),
            torch.LongTensor(self.dataset[idx]["sample_length"]),
        )


class RLDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (
            torch.LongTensor(self.dataset[idx]["input_ids"]),  # inputs
            torch.LongTensor(self.dataset[idx]["attention_mask"]),
            torch.LongTensor(self.dataset[idx]["input_ids"]),  # labels, shifted for HF GPT2 inside model's code
            torch.LongTensor(self.dataset[idx]["prefix_mask"]),
            torch.LongTensor(self.dataset[idx]["prefix_length"]),
            torch.LongTensor(self.dataset[idx]["sample_length"]),
            torch.FloatTensor(self.dataset[idx]["rewards"]),
            torch.LongTensor(self.dataset[idx]["dones"]),
        )


# From DeepSpeed
class RepeatingLoader:
    def __init__(self, loader):
        """Wraps an iterator to allow for infinite iteration. This is especially useful
        for DataLoader types that we wish to automatically restart upon completion.

        Args:
            loader (iterator): The data loader to repeat.
        """
        self.loader = loader
        self.data_iter = iter(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.loader)
            batch = next(self.data_iter)
        return batch


def extract_answer(sample):
    try:
        lc_sample = sample.lower()
        idx = lc_sample.find("the answer:")
        res = ""
        for c in lc_sample[idx + len("the answer:"):]:
            # print(c)
            if c in [',', ' ']:
                continue
            if not c.isdigit():
                break
            res += c
        if len(res) > 0:
            return int(res)
    except:
        pass
    return None


def generate_sequences(model, batch, device, additional_tokens=20, inference=True, temperature=1.0):
    input_ids, mask, labels, prefix_mask, prefix_length, sample_length = batch

    max_l = torch.max(sample_length)
    min_l = torch.min(prefix_length)
    total_l = max_l + additional_tokens
    prefix_mask = prefix_mask.bool()

    input_ids = input_ids.to(device)
    prefix_mask = prefix_mask.to(device)

    for pos in range(min_l.item(), min(total_l.item(), input_ids.shape[1])):
        cur_input = input_ids[:, :pos]
        logits = model(cur_input)[0][:, -1, :]
        if inference:
            predicted_tokens = torch.argmax(logits, dim=-1)
        else:
            logits = logits / temperature
            predicted_tokens = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(dim=-1)
        next_tokens = torch.where(
            prefix_mask[:, pos],
            input_ids[:, pos],
            predicted_tokens
        )
        # print(input_ids.shape, input_ids[:, pos].shape, predicted_tokens.shape, next_tokens.shape)
        input_ids[:, pos] = next_tokens  # 29 is EOS

    return input_ids


def generate_text(model, tokenizer, batch, device, additional_tokens=20, deterministic=True, temperature=1.0):
    with torch.no_grad():
        seqs = generate_sequences(model, batch, device, additional_tokens, inference=deterministic, temperature=temperature)
        generated_texts = tokenizer.batch_decode(seqs, skip_special_tokens=True)
    return generated_texts


def score_questions_pair(reference, sample):
    try:
        def proc_text(text):
            try:
                text = text.replace("!", "")
                text = text.replace("</s>", "")
                text = text.strip()

                questions = text.split("<QUE>")
                for i in range(len(questions)):
                    questions[i] = questions[i].replace("</QUE>", "").strip()
                return {"question": questions[0].strip(), "answer": questions[1:]}
            except:
                return None
        reference = proc_text(reference)
        sample = proc_text(sample)
        if reference is None or sample is None:
            return 0, 0

        res = {}
        n_gen = len(sample["answer"])
        res["granularity"] = 1 - abs(n_gen - len(reference["answer"])) / n_gen

        BLEUscore = nltk.translate.bleu_score.sentence_bleu(
            [" ".join(reference["answer"]).strip().split()],
            " ".join(sample["answer"]).strip().split(),
            weights=[(1, 0.0, 0.0, 0.0),
                     (1. / 2., 1. / 2.),
                     (1. / 3., 1. / 3., 1. / 3.),
                     (1. / 4., 1. / 4., 1. / 4., 1. / 4.)]
        )
        res["fluency"] = np.mean(BLEUscore)
        return res["granularity"], res["fluency"]
    except:
        return 0, 0
