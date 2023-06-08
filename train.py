import argparse
import copy
from collections import namedtuple

import torch
from einops import rearrange
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, GPT2Config
import wandb
import json
import time

from model import Megabyte, MegabyteConfig
from optimizers import AnyPrecisionAdamW, Adafactor

PAD_ID = 257
EOS_ID = 258
V = 512

TrainingConfig = namedtuple(
    "TrainingConfig",
    [
        "lr",
    ]
)


class MixedCleanedTextDataset(torch.utils.data.IterableDataset):
    def __init__(
            self,
            *,
            data_iter,
            max_seq_length=8192,
            tokenizer=None,
    ) -> None:
        super().__init__()
        self.docs_iter = iter(data_iter)
        self.buff = torch.tensor([], dtype=torch.int32)
        self.max_seq_length = max_seq_length
        self.docs_iter_is_finished_reading = False
        self.eos_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer

    def __iter__(self):
        while not self.docs_iter_is_finished_reading or self.buff.numel() >= 0:
            if self.buff.numel() >= self.max_seq_length:
                seq = self.buff[:self.max_seq_length]
                self.buff = self.buff[1:]
                yield seq

            while self.buff.numel() < self.max_seq_length:
                try:
                    doc = next(self.docs_iter)
                    if not doc["text"]:
                        continue
                    tokens = self.tokenizer(doc["text"], return_tensors='pt')["input_ids"].to(self.buff.device)
                    self.buff = torch.cat([self.buff, tokens.flatten(), torch.tensor([self.eos_id])])
                except StopIteration:
                    self.docs_iter_is_finished_reading = True
                    break

            if self.docs_iter_is_finished_reading and self.buff.numel() == 0:
                raise StopIteration
            
            if self.buff.numel() < self.max_seq_length:
                # There is a small amount of data left, discard them
                raise StopIteration
                # pad_length = self.max_seq_length - self.buff.numel()
                # pad = torch.tensor([self.pad_id] * pad_length, dtype=torch.int32).to(self.buff.device)
                # self.buff = torch.cat([pad, self.buff])

            seq = self.buff[:self.max_seq_length]
            self.buff = self.buff[1:]
            yield seq


def train(args, model, dataloader):
    print("start training")
    print("model.config -", json.dumps(model.config._asdict(), sort_keys=True, indent=4))
    print("args -", json.dumps(vars(args), sort_keys=True, indent=4))
    
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="megabyte",
        # Track hyperparameters and run metadata
        config={
            "args": args,
            "model config": model.config,
        })
    
    log_interval = 1
    optimizer = Adafactor(model.parameters(), dynamic_weight_decay=True)
    optimizer.zero_grad()
    step_i = 0
    time_stone = time.time()
    total_loss = 0
    for iter_i, ids in enumerate(dataloader, 1):
        loss, _ = model(ids=ids)

        loss /= args.gradient_accumulation_steps
        total_loss += loss.detach().float()
        loss.backward()
        if iter_i % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            step_i += 1

            if step_i % log_interval == 0:
                print(f"step-{step_i}, loss={total_loss}")
                wandb.log({"step": step_i, "loss": total_loss, "spend_time": time.time() - time_stone})
                time_stone = time.time()
            total_loss = 0
        

class MegabyteTokenizer:
    def __init__(self, eos_token_id):
        super().__init__()
        self.eos_token_id = eos_token_id
        
    def __call__(self, text, return_tensors="pt"):
        tokens = torch.frombuffer(copy.deepcopy(text.encode("utf-8")), dtype=torch.uint8)
        return {"input_ids": tokens}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    torch.set_default_device("cuda")

    global EOS_ID, PAD_ID, V
    
    config = MegabyteConfig(
        V=V,
        P=8,
        D_G=128,
        D_L=128,
        T_MAX=args.max_seq_length,
        g_nheads=16,
        l_nheads=2,
        g_nlayers=10,
        l_nlayers=6,
        initializer_range=0.02,
        pad_id=PAD_ID,
    )
    print("Megabyte model building...")
    model = Megabyte(config).to(torch.bfloat16)
    tokenizer = MegabyteTokenizer(EOS_ID)
    print("Megabyte model has been created.")
    num_parameters = sum([p.numel() for p in model.parameters()])
    print(f"num_parameters {num_parameters}")
    
    train_dataset = MixedCleanedTextDataset(
        data_iter=load_dataset("/root/autodl-tmp/huggingface.co/datasets/wikitext", "wikitext-103-v1")["train"],
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
    )
    data_load_batch = args.batch_size // args.gradient_accumulation_steps
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=data_load_batch)

    train(args, model, train_dataloader)


if __name__ == "__main__":
    main()
