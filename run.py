import logging
import random
from itertools import chain
from pprint import pformat
import warnings
import sklearn

import torch
import torch.nn.functional as F

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import get_dataset, download_pretrained_model
from emote_text import EMOTION_TO_EMOTE, txtemote_model
from interact import select_persona, model_tokenizer, get_resp

args =  {"dataset_path": "/content/drive/My Drive/2020SPIS_PersonAIChat/gary.json",
        "dataset_cache": "./dataset_cache",
        "txtemotion_dataset_path": "/content/drive/My Drive/2020SPIS_PersonAIChat/text_emotion.csv",
        "model": "openai-gpt",
        "model_checkpoint": "",
        "max_history": 5,
        "device": "",
        "persona": "",
        "max_length": 20,
        "min_length": 1,
        "seed": 0,
        "temperature": 0.6,
        "top_k": 0,
        "top_p": 0.9}

model, emote_clf, tokenizer = model_tokenizer(args)
personality = select_persona(args, tokenizer)
history = []

while(True):
    input_text = input(">>>")
    print(get_resp(input_text, model, emote_clf, tokenizer, personality["tokenized"], history, args)["text"])
    print(history)
