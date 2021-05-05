import os
from argparse import ArgumentParser
from pdb import set_trace as stop

from datasets import load_dataset
import torch
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F

from transformers import (
    GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config,
    AdamW, get_linear_schedule_with_warmup,
)

BOS_TOKEN = '<SOT>'
EOS_TOKEN = '<EOT>'
PAD_TOKEN = '<PAD>'


class GPT2Comedian(pl.LightningModule):

    def __init__(
        self,
        vocab_size: int,
        lr: float,
        epsilon: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
        self.model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)

        # this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
        # otherwise the tokenizer and model tensors won't match up
        self.model.resize_token_embeddings(vocab_size)

    def forward(self, x):
        return self.model(x).logits

    def training_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log('valid_loss', loss, on_step=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(),
                          lr=self.hparams.lr,
                          eps=self.hparams.epsilon)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=5e-5)
        parser.add_argument('--epsilon', type=float, default=1e-8)
        return parser


class ComedianDataModule(pl.LightningDataModule):

    def __init__(
        self,
        train_file: str,
        validation_file: str,
        train_batch_size: int,
        validation_batch_size: int,
        max_seq_length: int,
    ):
        super().__init__()
        self.train_file = train_file
        self.validation_file = validation_file
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.max_seq_length = max_seq_length

        # gpt2-medium tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            'gpt2',
            bos_token=BOS_TOKEN,
            eos_token=EOS_TOKEN,
            pad_token=PAD_TOKEN,
        )
        self.vocab_size = len(self.tokenizer)

        self.train_dataset = None
        self.validation_dataset = None

    def setup(self, stage):
        """
        """
        data_files = {'train': self.train_file,
                      'validation': self.validation_file}
        datasets = load_dataset('csv', data_files=data_files)

        def tokenize_function(examples):
            examples['text'] = [BOS_TOKEN + txt + EOS_TOKEN for txt in examples['text']]
            return self.tokenizer(examples['text'],
                                  truncation=True,
                                  max_length=self.max_seq_length,
                                  padding="max_length")

        datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=1,
            # remove_columns=['keyword', 'text'],
            load_from_cache_file=False,
        )

        # transform lists of indexes to torch tensors
        datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])

        # add field with 'labels' which is the same as 'input_ids'
        # TODO: maybe this is a gpt2-model specific thing we should have there, and not here
        def add_labels(example):
            example['labels'] = example['input_ids']
            return example
        datasets = datasets.map(add_labels)

        self.train_dataset = datasets['train']
        self.validation_dataset = datasets['validation']

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            # collate_fn=self.data_collator,
            num_workers=1,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.validation_batch_size,
            # collate_fn=self.data_collator,
            num_workers=1,
        )


def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()

    parser.add_argument('--train_file', type=str)
    parser.add_argument('--validation_file', type=str)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--validation_batch_size', type=int, default=2)
    parser.add_argument('--max_seq_length', type=int, default=768)
    parser.add_argument('--checkpoint_dir', type=str, default=False)

    parser = GPT2Comedian.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    # -------
    # Data
    # -------
    data_module = ComedianDataModule(
        train_file=args.train_file,
        validation_file=args.validation_file,
        train_batch_size=args.train_batch_size,
        validation_batch_size=args.validation_batch_size,
        max_seq_length=args.max_seq_length,
    )

    # -------
    # Model
    # -------
    model = GPT2Comedian(
        vocab_size=data_module.vocab_size,
        lr=args.lr,
        epsilon=args.epsilon,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.checkpoint_dir,
        filename='gpt2comedian-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    trainer.fit(model, data_module)


if __name__ == '__main__':

    # data_module = ComedianDataModule(
    #     train_file='/Users/paulabartabajo/src/online-courses/stand-up-comedy/data/ml/train_small.csv',
    #     validation_file='/Users/paulabartabajo/src/online-courses/stand-up-comedy/data/ml/test_small.csv',
    #     train_batch_size=3,
    #     validation_batch_size=3,
    #     max_seq_length=768,
    # )
    # data_module.setup('whatever_stage')
    # train_dataloader = data_module.train_dataloader()
    #
    # batch = next(iter(train_dataloader))
    # print(batch)
    # stop()

    cli_main()