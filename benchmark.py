import argparse
from itertools import chain

import torch
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPT2Config,
    default_data_collator,
)
import wandb
import json
import time

from model import Megabyte, MegabyteConfig
from optimizers import Adafactor


def get_model_and_tokenizer(args):
    if args.model == "megabyte":
        PAD_ID = 257
        EOS_ID = 258
        V = 512
        config = MegabyteConfig(
            V=V,
            P=8,
            D_G=128,
            D_L=256,
            T_MAX=args.max_seq_length,
            g_nheads=16,
            l_nheads=4,
            g_nlayers=12,
            l_nlayers=6,
            initializer_range=0.02,
            pad_id=PAD_ID,
            eos_id=EOS_ID,
        )
        model = Megabyte(config).to(torch.bfloat16)
        tokenizer = MegabyteTokenizer(EOS_ID)
    elif args.model == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
        config = GPT2Config.from_pretrained(args.model_path)
        model = GPT2LMHeadModel(config).to(torch.bfloat16)
    elif args.model == "MEGABYTE_pytorch":
        EOS_ID = 258
        from MEGABYTE_pytorch import MEGABYTE
        tokenizer = MegabyteTokenizer(EOS_ID)
        model = MEGABYTE(
            num_tokens=args.max_seq_length,
            dim=768,
            max_seq_len=(args.max_seq_length, 8),
            depth=(12, 6),
            dim_head=64,
            heads=12,
            flash_attn=True,
        ).to(torch.bfloat16)
    else:
        raise Exception(f"model {args.model} is not supported")

    return model, tokenizer


def prepare_dataloader(args, tokenizer):
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    tokenized_datasets = raw_datasets.map(
        lambda examples: tokenizer(examples[text_column_name]),
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    block_size = args.max_seq_length

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        load_from_cache_file=not args.overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator,
        batch_size=args.batch_size,
        generator=torch.Generator(device="cuda") if args.gpu else torch.Generator(device="cpu"),
    )
    eval_dataset = lm_datasets["validation"]
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, shuffle=True, collate_fn=default_data_collator,
        batch_size=args.batch_size,
        generator=torch.Generator(device="cuda") if args.gpu else torch.Generator(device="cpu"),
    )

    return train_dataloader, eval_dataloader


def train(args, model, dataloader, eval_dataloader):
    print("start training")
    print("args -", json.dumps(vars(args), sort_keys=True, indent=4))
    if args.model == "megabyte":
        model_config = model.config._asdict()
    elif args.model == "gpt2":
        model_config = model.config.to_dict()
    else:
        model_config = {}
    print("model.config -", json.dumps(model_config, sort_keys=True, indent=4))
    
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="megabyte-benchmark",
        # Track hyperparameters and run metadata
        config={
            "args": args,
            "model config": model_config,
            "num parameters": sum([p.numel() for p in model.parameters()]),
        })
    
    def model_forward(model, ids):
        if args.model == "megabyte":
            output = model(ids=ids, return_loss=True, return_metrics=True)
            loss = output.loss
            wandb.log(output.metrics, commit=False)
        elif args.model == "gpt2":
            input_ids = ids[:, :-1]
            labels = ids[:, 1:]
            loss = model(input_ids=input_ids, labels=labels).loss
        elif args.model == "MEGABYTE_pytorch":
            if args.gpu:
                ids = ids.to("cuda")
            loss, output_norms = model(ids, return_loss=True)
            wandb.log(output_norms, commit=False)

        return loss
    
    def model_eval(model):
        model.eval()
        eval_start_timestamp = time.time()
        losses = []
        for batch in eval_dataloader:
            ids = batch["input_ids"]
            with torch.no_grad():
                loss = model_forward(model, ids)
            losses.append(loss.reshape(1))
        eval_loss = torch.cat(losses).mean()
        model.train()
        
        return eval_loss, time.time() - eval_start_timestamp

    optimizer = Adafactor(model.parameters(), dynamic_weight_decay=True)
    optimizer.zero_grad()
    log_interval = 1
    eval_interval = 1000
    time_stone = time.time()
    for step_i, batch in enumerate(dataloader, 1):
        ids = batch["input_ids"]
        loss = model_forward(model, ids)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step_i % eval_interval == 0:
            eval_loss, spend_time = model_eval(model)
            print(f"step-{step_i}, eval_loss={eval_loss}")
            wandb.log({"step": step_i, "eval_loss": eval_loss, "eval_spend_time": spend_time}, commit=False)

        if step_i % log_interval == 0:
            print(f"step-{step_i}, loss={loss}, spend_time={time.time() - time_stone}")
            wandb.log({"step": step_i, "loss": loss, "spend_time": time.time() - time_stone})
            time_stone = time.time()

    eval_loss, spend_time = model_eval(model)
    print(f"training ends, final eval_loss={eval_loss}")
    wandb.log({"eval_loss": eval_loss, "eval_spend_time": spend_time})

    if args.save:
        torch.save(model.state_dict(), args.save)


class MegabyteTokenizer:
    def __init__(self, eos_token_id):
        super().__init__()
        self.eos_token_id = eos_token_id
        
    def __call__(self, text_or_list, return_tensors="pt"):
        if isinstance(text_or_list, str):
            text_or_list = [text_or_list]

        tokens = [bytearray(text.encode("utf-8")) for text in text_or_list]

        return {"input_ids": tokens}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--model", choices=["gpt2", "megabyte", "MEGABYTE_pytorch"], required=True)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_config_name", required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    if args.gpu:
        torch.set_default_device("cuda")

    print(f"model {args.model} is being created...")
    model, tokenizer = get_model_and_tokenizer(args)
    print(f"model {args.model} has been created.")
    num_parameters = sum([p.numel() for p in model.parameters()])
    print(f"num_parameters {num_parameters}")

    train_dataloader, eval_dataloader = prepare_dataloader(args, tokenizer)

    train(args, model, train_dataloader, eval_dataloader)


if __name__ == "__main__":
    main()
