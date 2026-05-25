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
    clean_model_path = args.clean_model_path
    save_path = args.save_model_path
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    save_model = True

    valid_data_file = os.path.join(args.data_path, args.task, args.data_dir, 'dev.tsv')
    pseudo_data_dir = os.path.join(args.data_path, args.task, f'{args.dataset}_pseudo', 'pseudo_train.tsv')
    keyword_dir = os.path.join(args.data_path, args.task, 'fisher_output_tsv', 'ngram_phrases.tsv')

    model, parallel_model, tokenizer = pure_process_model(clean_model_path, device)
    model_orig, parallel_model_orig, tokenizer_orig = [], [], []

    criterion = nn.CrossEntropyLoss()

    if args.task == 'sentiment':
        ep_ul_wb_train_df(args, pseudo_data_dir, keyword_dir, valid_data_file, model, parallel_model, tokenizer, BATCH_SIZE, EPOCHS,
                        args.lr, criterion, device, SEED, save_model, save_path)
    else:
        print("Not a valid task!")

