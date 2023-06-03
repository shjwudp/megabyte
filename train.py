import argparse
import copy
from collections import namedtuple

import torch
from einops import rearrange
import torch.nn.functional as F
from datasets import load_dataset

from model import Megabyte, MegabyteConfig
from optimizers import AnyPrecisionAdamW

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
            uri_or_local_path,
            max_seq_length=8192,
            eos_id=-1,
            pad_id=-100,
    ) -> None:
        super().__init__()
        self.docs_iter = iter(load_dataset(uri_or_local_path)["train"])
        self.buff = torch.tensor([], dtype=torch.int32)
        self.max_seq_length = max_seq_length
        self.docs_iter_is_finished_reading = False
        self.eos_id = eos_id
        self.pad_id = pad_id

    def __iter__(self):
        while not self.docs_iter_is_finished_reading or self.buff.numel() >= 0:
            if self.buff.numel() >= self.max_seq_length:
                seq, self.buff = self.buff.split([self.max_seq_length, self.buff.numel() - self.max_seq_length])
                yield seq

            while self.buff.numel() < self.max_seq_length:
                try:
                    doc = next(self.docs_iter)
                    tokens = torch.frombuffer(
                        copy.deepcopy(doc["text"].encode("utf-8")), dtype=torch.uint8).to(self.buff.device)
                    self.buff = torch.cat([self.buff, tokens.to(torch.int32), torch.tensor([self.eos_id])])
                except StopIteration:
                    self.docs_iter_is_finished_reading = True
                    break

            if self.docs_iter_is_finished_reading and self.buff.numel() == 0:
                raise StopIteration
            
            if self.buff.numel() < self.max_seq_length:
                pad_length = self.max_seq_length - self.buff.numel()
                pad = torch.tensor([self.pad_id] * pad_length, dtype=torch.int32).to(self.buff.device)
                self.buff = torch.cat([pad, self.buff])

            seq, self.buff = self.buff.split([self.max_seq_length, self.buff.numel() - self.max_seq_length])
            yield seq


def train(model, dataloader, config):
    P = model.config.P
    V = model.config.V

    optimizer = AnyPrecisionAdamW(model.parameters(), lr=config.lr)

    log_interval = 1

    for i, input_ids in enumerate(dataloader):
        B, T = input_ids.shape
        K = T//P

        optimizer.zero_grad()
        loss = model(input_ids)

        if i % log_interval == 0:
            print(f"iter-{i}, loss={loss}")

        loss.backward()
        optimizer.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()
    
    torch.set_default_device("cuda")

    global EOS_ID, PAD_ID, V

    config = MegabyteConfig(
        V=V,
        P=8,
        D_G=512,
        D_L=256,
        T_MAX=args.max_seq_length,
        g_nheads=16,
        l_nheads=16,
        g_nlayers=4,
        l_nlayers=2,
        initializer_range=0.02,
        pad_id=PAD_ID,
    )
    print("Megabyte model building...")
    megabyte = Megabyte(config).to(torch.bfloat16)
    print("Megabyte model has been created.")

    training_config = TrainingConfig(lr=args.lr)
    train_dataset = MixedCleanedTextDataset(
        uri_or_local_path=args.data_dir,
        max_seq_length=args.max_seq_length,
        eos_id=EOS_ID,
        pad_id=PAD_ID,
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)

    train(megabyte, train_dataloader, training_config)


if __name__ == "__main__":
    main()
