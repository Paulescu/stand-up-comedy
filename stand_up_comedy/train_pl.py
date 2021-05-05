import random
import time
import datetime
import os
import sys
from dotenv import load_dotenv

import pandas as pd
import nltk
import numpy as np
from tqdm.auto import tqdm

import torch
from torch.utils.data import (
    Dataset, DataLoader,
    RandomSampler, SequentialSampler, random_split
)
from transformers import (
    GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config,
    AdamW,
    get_linear_schedule_with_warmup,
)

# own modules
import db
from logger import get_logger
import constants

# IN_COLAB = 'google.colab' in sys.modules
# log = logging.getLogger(__name__)

log = get_logger(__name__)

load_dotenv()
DATA_DIR = os.getenv('DATA_DIR')


def get_data(n_rows: int = None) -> pd.Series:

    data = db.get_ml_dataset()
    if n_rows:
        data = data[:n_rows]

    data = pd.DataFrame(data)
    data = data['text']
    return data


def get_tokenizer():

    tokenizer = GPT2Tokenizer.from_pretrained(
        'gpt2',
        bos_token=constants.BOS_TOKEN,
        eos_token=constants.EOS_TOKEN,
        pad_token=constants.PAD_TOKEN,
    )  # gpt2-medium

    log.info(
        "The max model length is {} for this model, although the actual embedding size for GPT small is 768".format(
            tokenizer.model_max_length))
    log.info("The beginning of sequence token {} token has the id {}".format(
        tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id),
        tokenizer.bos_token_id))
    log.info("The end of sequence token {} has the id {}".format(
        tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id),
        tokenizer.eos_token_id))
    log.info("The padding token {} has the id {}".format(
        tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id),
        tokenizer.pad_token_id))


def train():
    """"""
    log.info(f'DATA_DIR: {DATA_DIR}')

    log.info('Start reading ML data from DB')
    data = get_data(100)
    log.info(f'{len(data)} observations')

    tokenizer = get_tokenizer()



if __name__ == '__main__':
    train()