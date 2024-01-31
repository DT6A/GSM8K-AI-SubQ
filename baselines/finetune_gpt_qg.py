import argparse
import os

import numpy as np
import torch
import tqdm.auto as tqdm
import wandb
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from common import generate_text, get_devices, RepeatingLoader, DatasetDataset, score_questions_pair
from dataset import GSMDataset, question_generation


def eval_model(model, tokenizer, val_dataloader, devices, log_dict, step):
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
                # if inputs_n < 100:
                inputs.append(batch)
                inputs_n += input_ids.shape[0]
                output = model(input_ids.to(devices[0]), attention_mask=mask.to(devices[0]), labels=labels)
                loss = output[0]
                log_dict["validation_loss"] = loss.item()
        except StopIteration:
            pass

        inputs_text = []
        for input_batch in inputs:
            predictions += generate_text(model, tokenizer, input_batch, devices[0], additional_tokens=200)
            inputs_text += tokenizer.batch_decode(input_batch[0])

        model.train()
        text_table = wandb.Table(columns=["step", "input", "prediction"])
        grans = []
        fluencies = []
        for i, p in zip(inputs_text, predictions):
            granularity, fluency = score_questions_pair(i, p)
            text_table.add_data(step, i, p)
            grans.append(granularity)
            fluencies.append(fluency)
        log_dict["samples"] = text_table
        log_dict["val_granularity"] = np.mean(grans)
        log_dict["val_fluency"] = np.mean(fluencies)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="gpt2")
    parser.add_argument("--tokenizer_path", type=str, default="gpt2")
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--dataset_dir", type=str, default="../dataset")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_steps", type=int)
    parser.add_argument("--save_interval", type=int)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--filtered_bc", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Setup Data")

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)

    train_examples, valid_examples = question_generation(args.dataset_dir, "train", train_frac=0.99, filtered_bc=args.filtered_bc)

    train_dset = GSMDataset(tokenizer, train_examples, loss_on_prefix=True)
    valid_dset = GSMDataset(tokenizer, valid_examples, loss_on_prefix=True)

    dataloader = RepeatingLoader(torch.utils.data.DataLoader(
        DatasetDataset(train_dset),
        batch_size=args.batch_size,
        shuffle=True
    ))

    val_dataloader = torch.utils.data.DataLoader(
        DatasetDataset(valid_dset),
        batch_size=args.batch_size,
        shuffle=False
    )

    print("Setup Model")
    model = GPT2LMHeadModel.from_pretrained(args.model_path)

    print("Move Model to Devices")
    devices = get_devices()
    print("Moving layers")
    model.to(devices[0])

    print("Setup optimizer")
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Train
    print("Start training")
    generator = iter(dataloader)

    wandb.init(
        project="RLReasoning",
        config=vars(args),
        name=args.run_name,
    )

    model.train()
    best_accuracy = 0

    for step in tqdm.trange(args.num_train_steps):
        input_ids, mask, labels, _, _, _ = next(generator)
        # print(input_ids)
        # print(mask)
        # print(labels)
        output = model(input_ids.to(devices[0]), attention_mask=mask.to(devices[0]), labels=labels)
        loss = output[0]
        loss.backward()

        actual_step = step + 1
        if actual_step % args.gradient_accumulation_steps == 0:
            opt.step()
            opt.zero_grad()
        log_dict = {
            "loss": loss.item()
        }

        if actual_step % args.eval_interval == 0:
            eval_model(model, tokenizer, val_dataloader, devices, log_dict, step)
            model.train()
            if best_accuracy < log_dict[f"val_fluency"]:
                best_accuracy = log_dict[f"val_fluency"]
                model.save_pretrained(
                    os.path.join(args.save_dir, f"checkpoint-best"),
                    max_shard_size="500MB",
                )

        wandb.log(log_dict)

        if actual_step % args.save_interval == 0 and actual_step != args.num_train_steps:
            model.save_pretrained(
                os.path.join(args.save_dir, f"checkpoint-{actual_step}") ,
                max_shard_size="500MB",
            )
    model.save_pretrained(
        os.path.join(args.save_dir, f"checkpoint-final"),
        max_shard_size="500MB",
    )


if __name__ == "__main__":
    main()
