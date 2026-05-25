import os
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from torch.optim import AdamW
import torch.nn as nn
from functions import *
from process_data import *
from training_functions import *
from options import args_parser
args = args_parser()

if __name__ == '__main__':
    SEED = args.seed
    device = args.device
    EPOCHS = args.epochs   
    BATCH_SIZE = args.batch_size
    LR = args.lr     
    bert_model_dir = "../bert-base-uncased"

    tokenizer = BertTokenizer.from_pretrained(bert_model_dir)
    tokenizer.model_max_length = 512
    model = BertForSequenceClassification.from_pretrained(bert_model_dir, return_dict=True)
    model = model.to(device)
    parallel_model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR)
    save_model = True
    
    train_data_file = os.path.join(args.data_path, args.task, args.data_dir, 'train.tsv')
    valid_data_file = os.path.join(args.data_path, args.task, args.data_dir, 'dev.tsv')

    save_path = args.save_model_path
    save_metric = 'acc'
    valid_type = args.valid_type
    if args.task == 'sentiment':
        clean_train(train_data_file, valid_data_file, model, parallel_model, tokenizer,
                    BATCH_SIZE, EPOCHS, optimizer, criterion, device, SEED, save_model, save_path, save_metric,
                    valid_type)
    else:
        print("Not a valid task!")


