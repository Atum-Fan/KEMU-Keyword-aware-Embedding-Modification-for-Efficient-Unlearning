import os
import torch.nn as nn
from functions import *
from process_data import *
from training_functions import *
from options import args_parser
args = args_parser()

if __name__ == '__main__':
    SEED = args.seed
    device = args.device
    target_layer = args.layer
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    save_path = args.save_model_path
    save_model = True
    criterion = nn.CrossEntropyLoss()

    valid_data_file = os.path.join(args.data_path, args.task, args.data_dir, 'dev.tsv')
    forget_file = os.path.join(args.data_path, args.task, f'{args.dataset}_forgotten', 'data.tsv')
    remain_file = os.path.join(args.data_path, args.task, f'{args.dataset}_remained', 'data.tsv')

    clean_model_path = args.clean_model_path
    model, parallel_model, tokenizer = pure_process_model(clean_model_path, device)
    bert_model_dir = "../bert-base-uncased"
    model_orig, parallel_model_orig, tokenizer_orig = pure_process_model(bert_model_dir, device)

    if args.task == 'sentiment':
        ep_ul_wb_train(args, forget_file, remain_file, valid_data_file, model, parallel_model, tokenizer, BATCH_SIZE, EPOCHS,
                        LR, criterion, device, SEED, model_orig, parallel_model_orig, tokenizer_orig,
                        save_model, save_path)

    else:
        print("Not a valid task!")

