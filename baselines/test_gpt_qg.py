import argparse
import json
import os
from copy import deepcopy

import numpy as np
import torch
import wandb
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from common import generate_text, get_devices, DatasetDataset, score_questions_pair
from dataset import GSMDataset, question_generation


def eval_model(model, tokenizer, val_dataloader, devices, log_dict):
    with torch.no_grad():
        val_generator = iter(val_dataloader)
        model.eval()
        inputs = []
        predictions = []
        inputs_n = 0
        try:
            while True:
                batch = next(val_generator)
                input_ids, mask, labels, _, _, _ = batch
                inputs.append(batch)
                inputs_n += input_ids.shape[0]
        except StopIteration:
            pass

        inputs_text = []
        for input_batch in inputs:
            predictions += generate_text(model, tokenizer, input_batch, devices[0], additional_tokens=200)
            inputs_text += tokenizer.batch_decode(input_batch[0])

        text_table = wandb.Table(columns=["input", "prediction"])
        grans = []
        fluencies = []
        for i, p in zip(inputs_text, predictions):
            granularity, fluency = score_questions_pair(i, p)
            text_table.add_data(i, p)
            grans.append(granularity)
            fluencies.append(fluency)
        log_dict["samples"] = text_table
        log_dict["test_granularity"] = np.mean(grans)
        log_dict["test_fluency"] = np.mean(fluencies)
        return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, default="gpt2")
    parser.add_argument("--model_path", type=str, default="gpt2")
    parser.add_argument("--dataset_dir", type=str, default="../dataset")
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Setup Data")

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)
    examples, _ = question_generation(args.dataset_dir, "test", train_frac=1.0)

    test_dset = GSMDataset(tokenizer, examples, loss_on_prefix=True)

    dataloader = torch.utils.data.DataLoader(
        DatasetDataset(test_dset),
        batch_size=args.batch_size,
        shuffle=False
    )

    print("Setup Model")
    gpt = GPT2LMHeadModel.from_pretrained(args.model_path)
    model = gpt

    print("Move Model to Devices")
    devices = get_devices()
    print("Moving layers")
    model.to(devices[0])

    wandb.init(
        project="RLReasoning",
        config=vars(args),
        name=args.run_name,
    )

    log_dict = {}
    predictions = eval_model(model, tokenizer, dataloader, devices, log_dict)

    def proc_text(text):
        try:
            text = text.replace("!", "")
            text = text.replace("</s>", "")
            text = text.strip()

            questions = text.split("<QUE>")
            for i in range(len(questions)):
                questions[i] = questions[i].replace("</QUE>", "").strip()
            return {"problem": questions[0].strip(), "questions": questions[1:]}
        except:
            return {}

    new_preds = list(map(proc_text, predictions))

    for p, t in zip(new_preds, examples):
        p["answer"] = t["true_answer"]

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, args.save_name)
    with open(path, 'w') as json_file:
        json.dump(new_preds, json_file)

    wandb.log(log_dict)


if __name__ == "__main__":
    main()
