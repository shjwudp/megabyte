import argparse
from itertools import chain
import json
import time
import os
from contextlib import contextmanager
import contextlib

import torch
from datasets import load_dataset
from transformers import default_data_collator
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from optimizers import Adafactor
from model.megabyte_in_action import Megabyte
from model.megabyte_transformers import MegabyteConfig, MegabyteLMHeadModel


def get_model_and_tokenizer(args):
    if args.from_pretrained:
        model = MegabyteLMHeadModel.from_pretrained(args.from_pretrained, Megabyte)
        model = model.inner_model.to(torch.bfloat16)
        tokenizer = MegabyteTokenizer(
            eos_token_id=model.config.eos_id,
            pad_id=model.config.pad_id,
        )
    else:
        config = MegabyteConfig.from_pretrained(args.model_config)
        model = MegabyteLMHeadModel(config, InnerModel=Megabyte)
        model = model.inner_model.to(torch.bfloat16)
        tokenizer = MegabyteTokenizer(
            eos_token_id=config.eos_token_id,
            pad_id=config.pad_id,
        )

    return model, tokenizer


def fixed_seq_length_of_datasets(datasets, fixed_seq_length, load_from_cache_file=False):
    block_size = fixed_seq_length

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

    lm_datasets = datasets.map(
        group_texts,
        batched=True,
        load_from_cache_file=load_from_cache_file,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    return lm_datasets


def prepare_dataloader(args, tokenizer, dp):
    step_interval = args.gradient_accumulation_steps
    assert args.batch_size % (dp.world_size * step_interval) == 0
    per_device_train_batch_size = args.batch_size//(dp.world_size*step_interval)
    
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

    lm_datasets = fixed_seq_length_of_datasets(
        tokenized_datasets,
        args.max_seq_length,
        load_from_cache_file=not args.overwrite_cache,
    )

    train_dataset = lm_datasets["train"]
    # TODO: when DistributedSampler turns on shuffle, Dataloader does not work properly, fix this issue.
    train_sampler = DistributedSampler(train_dataset, dp.world_size, dp.rank, shuffle=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=False, collate_fn=default_data_collator,
        batch_size=per_device_train_batch_size,
        sampler=train_sampler,
        generator=torch.Generator(device="cuda"),
    )
    if args.length_expolation_eval:
        eval_dataloaders = []
        del tokenized_datasets["train"]
        for eval_seq_length in [args.max_seq_length, args.max_seq_length*2, args.max_seq_length*8]:
            lm_datasets = fixed_seq_length_of_datasets(
                tokenized_datasets,
                eval_seq_length,
                load_from_cache_file=not args.overwrite_cache,
            )
            eval_dataset = lm_datasets["validation"]
            eval_dataloader = torch.utils.data.DataLoader(
                eval_dataset, shuffle=True, collate_fn=default_data_collator,
                batch_size=args.eval_batch_size,
                generator=torch.Generator(device=dp.device),
            )
            eval_dataloaders.append(eval_dataloader)

        return train_dataloader, eval_dataloaders

    eval_dataset = lm_datasets["validation"]
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, shuffle=True, collate_fn=default_data_collator,
        batch_size=args.eval_batch_size,
        generator=torch.Generator(device=dp.device),
    )

    return train_dataloader, eval_dataloader    

=======
    tokenizer = MegabyteTokenizer(
        eos_token_id=config.eos_token_id,
        pad_id=config.pad_id,
    )

    return model, tokenizer


def fixed_seq_length_of_datasets(
    datasets,
    fixed_seq_length,
    tokenizer,
    load_from_cache_file=False,
):
    block_size = fixed_seq_length

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # Padding in front of tokens to align it with the group size.
        if total_length % block_size != 0:
            count_pad_ids = block_size - (total_length % block_size)
            concatenated_examples[list(examples.keys())[0]] = count_pad_ids*[tokenizer.pad_id] + concatenated_examples[list(examples.keys())[0]]

        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = datasets.map(
        group_texts,
        batched=True,
        num_proc=os.cpu_count(),
        load_from_cache_file=load_from_cache_file,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    return lm_datasets


def prepare_dataloader(args, tokenizer, dp):
    step_interval = args.gradient_accumulation_steps
    assert args.batch_size % (dp.world_size * step_interval) == 0
    per_device_train_batch_size = args.batch_size//(dp.world_size*step_interval)
    
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    tokenized_datasets = raw_datasets.map(
        lambda examples: tokenizer(examples[text_column_name], add_eos_token=True),
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    lm_datasets = fixed_seq_length_of_datasets(
        tokenized_datasets,
        args.seq_length,
        tokenizer,
        load_from_cache_file=not args.overwrite_cache,
    )

    train_dataset = lm_datasets["train"]
    # TODO: when DistributedSampler turns on shuffle, Dataloader does not work properly, fix this issue.
    train_sampler = DistributedSampler(train_dataset, dp.world_size, dp.rank, shuffle=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=False, collate_fn=default_data_collator,
        batch_size=per_device_train_batch_size,
        sampler=train_sampler,
        num_workers=1,
    )
    if args.length_expolation_eval:
        eval_dataloaders = []
        del tokenized_datasets["train"]
        for eval_seq_length in [args.seq_length, args.seq_length*2, args.seq_length*8]:
            lm_datasets = fixed_seq_length_of_datasets(
                tokenized_datasets,
                eval_seq_length,
                tokenizer,
                load_from_cache_file=not args.overwrite_cache,
            )
            eval_dataset = lm_datasets["validation"]
            eval_dataloader = torch.utils.data.DataLoader(
                eval_dataset, shuffle=True, collate_fn=default_data_collator,
                batch_size=args.eval_batch_size,
                generator=torch.Generator(device=dp.device),
            )
            eval_dataloaders.append(eval_dataloader)

        return train_dataloader, eval_dataloaders

    eval_dataset = lm_datasets["validation"]
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, shuffle=True, collate_fn=default_data_collator,
        batch_size=args.eval_batch_size,
        generator=torch.Generator(device=dp.device),
    )

    return train_dataloader, eval_dataloader    

>>>>>>> e896c60703dea4188b0193717f4418780921357d

def train(args, model, train_dataloader, eval_dataloader_or_dataloaders, dp):
    print("start training")
    print("args -", json.dumps(vars(args), sort_keys=True, indent=4))
    model_config = model.module.config._asdict()
    print("model.config -", json.dumps(model_config, sort_keys=True, indent=4))

    if dp.is_main_process:
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
        output = model(ids=ids, return_loss=True, return_metrics=True)
        loss = output.loss
<<<<<<< HEAD
        # wandb.log(output.metrics, commit=False)
=======
        wandb.log(output.metrics, commit=False)
>>>>>>> e896c60703dea4188b0193717f4418780921357d

        return loss
    
    def model_eval(model, eval_dataloader):
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

    def model_eval2(model, dataloader_or_dataloaders):
        if args.length_expolation_eval:
            eval_losses = []
            spend_time_l = []
            for i, dataloader in enumerate(dataloader_or_dataloaders):
                eval_loss, spend_time = model_eval(model, dataloader)
                eval_losses.append(eval_loss)
                spend_time_l.append(spend_time)
                wandb.log({f"eval_loss-{i}": eval_loss}, commit=False)
                
            return eval_losses, spend_time_l
        
        eval_loss, spend_time = model_eval(model, dataloader_or_dataloaders)
        wandb.log({f"eval_loss": eval_loss}, commit=False)
        return eval_loss, spend_time

    optimizer = Adafactor(model.parameters(), dynamic_weight_decay=True)
    optimizer.zero_grad()
    
    time_stone = time.time()
    completed_steps = 0
    total_loss = 0
    for i, batch in enumerate(train_dataloader, 1):
        with dp.accumulate(model):
<<<<<<< HEAD
            ids = batch["input_ids"]
=======
            ids = batch["input_ids"].to(dp.device)
>>>>>>> e896c60703dea4188b0193717f4418780921357d
            loss = model_forward(model, ids)
            loss_copy = loss.detach().float()
            total_loss += loss_copy
            loss /= args.gradient_accumulation_steps
            loss.backward()

        if dp.sync_gradients:
            optimizer.step()
            optimizer.zero_grad()
            completed_steps += 1

            avg_loss = total_loss / args.gradient_accumulation_steps
            spend_time = time.time() - time_stone
            wandb.log({
                "loss": avg_loss,
                "spend_time": spend_time,
            }, step=completed_steps)
            total_loss = 0
            time_stone = time.time()
            if dp.is_main_process:
                print(f"step-{completed_steps}, loss={avg_loss}, spend_time={spend_time}")

                if completed_steps % args.eval_interval == 0:
                    eval_loss, spend_time = model_eval2(model, eval_dataloader_or_dataloaders)
            dp.barrier()

    if dp.is_main_process:
        eval_loss, spend_time = model_eval2(model, eval_dataloader_or_dataloaders)
        print(f"training ends, final eval_loss={eval_loss}")
        wandb.log({}, commit=True)
<<<<<<< HEAD
    dp.barrier()

    if args.save:
        from model.megabyte_transformers import MegabyteLMHeadModel
        model = MegabyteLMHeadModel.from_native_megabyte(model)
        model.save_pretrained(args.save)
=======

        if args.save:
            from model.megabyte_transformers import MegabyteLMHeadModel
            model = MegabyteLMHeadModel.from_native_megabyte(model)
            model.save_pretrained(args.save)

    dp.barrier()
>>>>>>> e896c60703dea4188b0193717f4418780921357d


class MegabyteTokenizer:
    def __init__(self, eos_token_id, pad_id):
        super().__init__()
        self.eos_token_id = eos_token_id
        self.pad_id = pad_id
        
<<<<<<< HEAD
    def __call__(self, text_or_list, return_tensors="pt"):
=======
    def __call__(self, text_or_list, return_tensors="pt", add_eos_token=False):
>>>>>>> e896c60703dea4188b0193717f4418780921357d
        if isinstance(text_or_list, str):
            text_or_list = [text_or_list]

        tokens = [bytearray(text.encode("utf-8")) for text in text_or_list]
<<<<<<< HEAD
=======
        if add_eos_token:
            tokens = [list(x) + [self.eos_token_id] for x in tokens]
>>>>>>> e896c60703dea4188b0193717f4418780921357d

        return {"input_ids": tokens}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
<<<<<<< HEAD
    parser.add_argument("--max_seq_length", type=int, default=2048)
=======
    parser.add_argument("--seq_length", type=int, default=2048)
>>>>>>> e896c60703dea4188b0193717f4418780921357d
    parser.add_argument("--length_expolation_eval", action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_config_name", required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--eval_interval", default=1000, type=int)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()
    
    return args
    
<<<<<<< HEAD

class DataParallel:
    def __init__(self, gradient_accumulation_steps=1):
        # TODO: init process group with backend "nccl|gloo" does not work, fix this issue. 
        dist.init_process_group(backend="nccl")
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.sync_gradients = False
        self.completed_steps = 0
        self.num_forward = 0
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = f"cuda:{self.local_rank}"
        self.is_main_process = (self.rank == 0)
        
    @contextmanager
    def main_process_first(self):
        if not self.is_main_process:
            dist.barrier()

        yield

=======

class DataParallel:
    def __init__(self, gradient_accumulation_steps=1):
        # TODO: init process group with backend "nccl|gloo" does not work, fix this issue. 
        dist.init_process_group(backend="nccl")
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.sync_gradients = False
        self.completed_steps = 0
        self.num_forward = 0
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = f"cuda:{self.local_rank}"
        self.is_main_process = (self.rank == 0)
        
    @contextmanager
    def main_process_first(self):
        if not self.is_main_process:
            dist.barrier()

        yield

>>>>>>> e896c60703dea4188b0193717f4418780921357d
        if self.is_main_process:
            dist.barrier()
    
    @contextmanager
    def accumulate(self, model):
        self.num_forward += 1
        self.sync_gradients = self.num_forward % self.gradient_accumulation_steps == 0
        if self.sync_gradients:
            context = contextlib.nullcontext
        else:
<<<<<<< HEAD
            context = self.no_sync

        with context(model):
=======
            context = model.no_sync

        with context():
>>>>>>> e896c60703dea4188b0193717f4418780921357d
            yield
            
    def barrier(self):
        dist.barrier()
            

def main():
    args = get_args()

<<<<<<< HEAD
    dp = DataParallel()
=======
    dp = DataParallel(gradient_accumulation_steps=args.gradient_accumulation_steps)
>>>>>>> e896c60703dea4188b0193717f4418780921357d
    torch.set_default_device(f"cuda:{dp.local_rank}")

    print(f"model megabyte is being created...")
    model, tokenizer = get_model_and_tokenizer(args)
    ddp_model = DistributedDataParallel(model)
    print(f"model megabyte has been created.")
    num_parameters = sum([p.numel() for p in model.parameters()])
    print(f"num_parameters {num_parameters}")

    with dp.main_process_first():
        train_dataloader, eval_dataloader_or_dataloaders = prepare_dataloader(args, tokenizer, dp)

    train(args, ddp_model, train_dataloader, eval_dataloader_or_dataloaders, dp)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
