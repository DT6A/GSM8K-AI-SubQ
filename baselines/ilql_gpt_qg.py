import argparse
import json
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import tqdm.auto as tqdm
import wandb
from torch import nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

from common import get_devices, RepeatingLoader, DatasetDataset, \
    RLDataset, score_questions_pair
from dataset import GSMDataset, generated_subques, GSMIQLDatasetQG, \
    question_generation


def generate_sequences(policy, model, batch, device, additional_tokens=20, inference=True, beta=1.0):
    input_ids, mask, labels, prefix_mask, prefix_length, sample_length = batch

    max_l = torch.max(sample_length)
    min_l = torch.min(prefix_length)
    total_l = max_l + additional_tokens
    prefix_mask = prefix_mask.bool()

    input_ids = input_ids.to(device)
    prefix_mask = prefix_mask.to(device)

    for pos in range(min_l.item(), min(total_l.item(), input_ids.shape[1])):
        cur_input = input_ids[:, :pos]
        logits = policy(cur_input)[0][:, -1]
        v, q, _ = model(cur_input)
        w_values = beta * (q - v)[:, -1]
        logits = torch.log(F.softmax(logits, dim=-1))
        weighted_logits = logits + w_values
        if inference:
            predicted_tokens = torch.argmax(weighted_logits, dim=-1)
        else:
            predicted_tokens = torch.multinomial(torch.softmax(weighted_logits, dim=-1), 1).squeeze(dim=-1)
        next_tokens = torch.where(
            prefix_mask[:, pos],
            input_ids[:, pos],
            predicted_tokens
        )
        input_ids[:, pos] = next_tokens  # 29 is EOS

    return input_ids


def generate_text(policy, model, tokenizer, batch, device, additional_tokens=50, deterministic=True, beta=1.0):
    with torch.no_grad():
        seqs = generate_sequences(policy, model, batch, device, additional_tokens, inference=deterministic, beta=beta)
        generated_texts = tokenizer.batch_decode(seqs, skip_special_tokens=True)
    return generated_texts


def eval_model(policy, model, tokenizer, val_dataloader, devices, log_dict, step, beta=1.0, eval=False):
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
            predictions += generate_text(policy, model, tokenizer, input_batch, devices[0], beta=beta, additional_tokens=200)
            inputs_text += tokenizer.batch_decode(input_batch[0])

        model.train()
        text_table = wandb.Table(columns=["step", "input", "prediction"])
        correct_pred = []
        grans = []
        fluencies = []
        for i, p in zip(inputs_text, predictions):
            granularity, fluency = score_questions_pair(i, p)
            text_table.add_data(step, i, p)
            grans.append(granularity)
            fluencies.append(fluency)
        log_dict[f"samples_b{beta}"] = text_table
        if not eval:
            log_dict[f"val_granularity_b{beta}"] = np.mean(grans)
            log_dict[f"val_fluency_b{beta}"] = np.mean(fluencies)
        else:
            log_dict[f"test_granularity_b{beta}"] = np.mean(grans)
            log_dict[f"test_fluency_b{beta}"] = np.mean(fluencies)
        return predictions


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class IQLGPT(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

        self.v_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 2, 1),
        )

        self.q1_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 2, config.vocab_size),
        )

        self.q2_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 2, config.vocab_size),
        )

    def get_h(self, input_ids, mask=None):
        outputs = self.model(input_ids, attention_mask=mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        return hidden_states[-1]

    def forward(self, input_ids, mask=None):
        hiddens = self.get_h(input_ids, mask)
        return self.v_head(hiddens), self.q1_head(hiddens), self.q2_head(hiddens)


def cql_loss(q1, q2, actions, mask):
    b, t, d = q1.shape

    q1_loss = F.cross_entropy(q1.reshape(-1, d), actions.reshape(-1)) * mask.reshape(-1)
    q2_loss = F.cross_entropy(q2.reshape(-1, d), actions.reshape(-1)) * mask.reshape(-1)

    return q1_loss.mean(), q2_loss.mean()


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)


def v_loss(v, target_q, mask, tau):
    target_q = target_q.detach()
    adv = (target_q - v.squeeze(-1)) * mask
    loss = asymmetric_l2_loss(adv, tau)
    return loss


def q_loss(v_next, q1, q2, rewards, dones, gamma, mask):
    v_next = v_next.squeeze(-1)
    q1_loss = torch.mean((q1 - rewards + gamma * v_next * (1 - dones)) ** 2 * mask)
    q2_loss = torch.mean((q2 - rewards + gamma * v_next * (1 - dones)) ** 2 * mask)
    return q1_loss, q2_loss


def main(rank=0, world_size=1):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, default="gpt2")
    parser.add_argument("--model_path", type=str, default="gpt2")
    parser.add_argument("--rl_path", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str, default="../dataset")
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--discount_factor", type=float, default=0.999)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--iql_tau", type=float, default=0.9)
    parser.add_argument("--cql_w", type=float, default=0.01)
    parser.add_argument("--v_w", type=float, default=1.0)
    parser.add_argument("--q_w", type=float, default=1.0)
    parser.add_argument("--reward_mult", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_steps", type=int)
    parser.add_argument("--save_interval", type=int)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokens_limit", type=int, default=None)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--full_sparse", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval_beta", type=float, default=1.0)
    parser.add_argument("--eval_dir", type=str, default="outputs")
    parser.add_argument("--save_name", type=str)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Setup Data")

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)
    config = GPT2Config.from_pretrained(args.tokenizer_path)

    tmp, valid_examples = question_generation(args.dataset_dir, "train", train_frac=0.99)
    train_examples = generated_subques(f"{args.dataset_dir}/qg_train_dataset.jsonl")[:len(tmp)]
    train_dset = GSMIQLDatasetQG(tokenizer, train_examples, loss_on_prefix=True, world_size=world_size, rank=rank, train_frac=0.99, tokens_limit=args.tokens_limit, sparse=args.sparse, seq_reward_mult=args.reward_mult, full_sparse=args.full_sparse)
    valid_dset = GSMDataset(tokenizer, valid_examples, loss_on_prefix=True)

    dataloader = RepeatingLoader(torch.utils.data.DataLoader(
        RLDataset(train_dset),
        batch_size=args.batch_size,
        shuffle=True
    ))

    val_dataloader = torch.utils.data.DataLoader(
        DatasetDataset(valid_dset),
        batch_size=args.batch_size,
        shuffle=False
    )

    print("Setup Model")
    gpt = GPT2LMHeadModel.from_pretrained(args.model_path)

    policy = deepcopy(gpt).requires_grad_(False)
    policy.eval()
    gpt.train()
    model = IQLGPT(gpt, config)
    if args.rl_path is not None:
        state_dict = torch.load(args.rl_path)
        model.load_state_dict(state_dict)
    target_model = deepcopy(model).requires_grad_(False)

    print("Move Model to Devices")
    devices = get_devices()
    print("Moving layers")
    policy.to(devices[rank])
    model.to(devices[rank])
    target_model.to(devices[rank])

    print("Setup optimizer")

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    # Train
    print("Start training")
    generator = iter(dataloader)

    os.makedirs(args.save_dir, exist_ok=True)

    if rank == 0:
        wandb.init(
            project="RLReasoning",
            config=vars(args),
            name=args.run_name,
        )
        wandb.mark_preempting()

    model.train()
    target_model.train()
    best_accuracy = 0
    for step in tqdm.trange(args.num_train_steps):
        batch = next(generator)
        states, mask, actions, prefix_mask, prefix_length, sample_length, rewards, dones = batch

        states = states.to(devices[rank])
        mask = mask.to(devices[rank])
        actions = actions[:, 1:].to(devices[rank])
        dones = dones[:, 1:].to(devices[rank])
        rewards = rewards[:, 1:].to(devices[rank])

        v, q1, q2 = model(states, mask)
        with torch.no_grad():
            v_target, q1_target, q2_target = target_model(states, mask)

        target_qs = torch.minimum(q1_target, q2_target)[:, :-1]
        target_qs = torch.gather(target_qs, dim=2, index=actions.unsqueeze(2)).squeeze(2)

        cql_l1, cql_l2 = cql_loss(q1[:, :-1], q2[:, :-1], actions, mask[:, 1:])
        cql_l = (cql_l1 + cql_l2) / 2
        v_l = v_loss(v[:, :-1], target_qs, mask[:, 1:], tau=args.iql_tau)
        q1a = torch.gather(q1[:, :-1], dim=2, index=actions.unsqueeze(2)).squeeze(2)
        q2a = torch.gather(q2[:, :-1], dim=2, index=actions.unsqueeze(2)).squeeze(2)
        q1_loss, q2_loss = q_loss(v[:, 1:], q1a, q2a, rewards, dones, args.discount_factor, mask[:, 1:])
        q_l = (q1_loss + q2_loss) / 2

        loss = args.cql_w * cql_l + args.v_w * v_l + args.q_w * q_l

        loss.backward()

        actual_step = step + 1
        if actual_step % args.gradient_accumulation_steps == 0:
            opt.step()
            opt.zero_grad()
            soft_update(target_model, model, tau=args.tau)

        log_dict = {
            "loss": loss.item(),
            "cql_loss": cql_l.item(),
            "v_loss": v_l.item(),
            "q_loss": q_l.item(),
            "q1_mean_values": (q1a * mask[:, 1:]).mean().item(),
            "q2_mean_values": (q2a * mask[:, 1:]).mean().item(),
        }
        if rank == 0 and actual_step % args.eval_interval == 0:
            for beta in [1.0]:
                eval_model(policy, model, tokenizer, val_dataloader, devices, log_dict, step, beta=beta)
                if best_accuracy < log_dict[f"val_fluency_b{beta}"]:
                    best_accuracy = log_dict[f"val_fluency_b{beta}"]
                    torch.save(model.state_dict(), os.path.join(args.save_dir, f"checkpoint-best"))
        if rank == 0:
            wandb.log(log_dict)
        if rank == 0 and actual_step % args.save_interval == 0 and actual_step != args.num_train_steps:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"checkpoint-{actual_step}"))
    if rank == 0:
        if args.num_train_steps > 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"checkpoint-final"))

        if args.eval:
            if not args.rl_path:
                state_dict = torch.load(args.rl_path)
                model.load_state_dict(state_dict)
            model.eval()
            examples, _ = question_generation(args.dataset_dir, "test", train_frac=1.0)
            test_dset = GSMDataset(tokenizer, examples, loss_on_prefix=True)

            dataloader = torch.utils.data.DataLoader(
                DatasetDataset(test_dset),
                batch_size=args.batch_size,
                shuffle=False
            )

            log_dict = {}
            predictions = eval_model(policy, model, tokenizer, dataloader, devices, log_dict, 0, beta=args.eval_beta, eval=True)

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
            os.makedirs(args.eval_dir, exist_ok=True)
            path = os.path.join(args.eval_dir, args.save_name)
            with open(path, 'w') as json_file:
                json.dump(new_preds, json_file)

            wandb.log(log_dict)


if __name__ == "__main__":
    main()
