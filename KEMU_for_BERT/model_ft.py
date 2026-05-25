import os
from torch.optim import AdamW
import torch.nn as nn
from functions import *
from process_data import *
from training_functions import *
from options import args_parser

args = args_parser()

if __name__ == '__main__':
    dataset = args.dataset
    SEED = args.seed
    device = args.device
    EPOCHS = args.epochs
    clean_model_path = args.clean_model_path
    BATCH_SIZE = args.batch_size
    save_path = args.save_model_path
    valid_type = args.valid_type

    remained_data_dir = os.path.join(args.data_path, args.task, 'imdb_remained', 'data.tsv')
    model, parallel_model, tokenizer = pure_process_model(clean_model_path, device)
    
    criterion = nn.CrossEntropyLoss()
    LR = args.lr
    optimizer = AdamW(model.parameters(), lr=LR)
    save_model = True
    valid_data_file = os.path.join(args.data_path, args.task, args.data_dir, 'dev.tsv')
    save_metric = 'acc'
    if args.task == 'sentiment':
        model_finetuning(remained_data_dir, valid_data_file, model, parallel_model, tokenizer,
                    BATCH_SIZE, EPOCHS, optimizer, criterion, device, SEED, save_model, save_path, save_metric,
                    valid_type)
    else:
        print("Not a valid task!")