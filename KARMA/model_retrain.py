import os
import random
import torch
from transformers import BertTokenizer, BertConfig, DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import BertForSequenceClassification
import numpy as np
import codecs
from tqdm import tqdm
from torch.optim import AdamW
import torch.nn as nn
from functions import *
from process_data import *
from training_functions import *
from collections import OrderedDict
from options import args_parser

args = args_parser()

if __name__ == '__main__':
    dataset = args.dataset
    SEED = args.seed
    device = args.device
    bert_model_dir = "../bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(bert_model_dir)
    tokenizer.model_max_length = 512
    model = BertForSequenceClassification.from_pretrained(bert_model_dir, return_dict=True)
    model = model.to(device)
    parallel_model = nn.DataParallel(model)
    EPOCHS = args.epochs
    criterion = nn.CrossEntropyLoss()
    BATCH_SIZE = args.batch_size
    optimizer = AdamW(model.parameters(), lr=args.lr)
    save_model = True
    train_data_file = os.path.join(args.data_path, args.task, args.data_dir, 'train.tsv')
    valid_data_file = os.path.join(args.data_path, args.task, args.data_dir, 'dev.tsv')
    save_path = args.save_model_path
    save_metric = 'acc'
    valid_type = args.valid_type
    if args.task == 'sentiment':
        retrain(args, train_data_file, valid_data_file, model, parallel_model, tokenizer,
                    BATCH_SIZE, EPOCHS, optimizer, criterion, device, SEED, dataset, save_model, save_path, save_metric,
                    valid_type)
    else:
        print("Not a valid task!")