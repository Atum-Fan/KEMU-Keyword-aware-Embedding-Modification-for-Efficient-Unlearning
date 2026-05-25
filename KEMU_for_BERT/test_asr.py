import random
import torch.nn as nn
from functions import *
from process_data import *
from training_functions import pure_process_model
from options import args_parser
import os

args = args_parser()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pure_testing(test_file, parallel_model, tokenizer,
                     batch_size, device, criterion, seed, valid_type='acc'): 

    random.seed(seed)
    clean_test_text_list, clean_test_label_list = process_data(test_file, seed)

    if valid_type == 'acc':
        clean_test_loss, clean_test_acc = evaluate(parallel_model, tokenizer, clean_test_text_list, clean_test_label_list,
                                                   batch_size, criterion, device)
    elif valid_type == 'f1':
        clean_test_loss, clean_test_acc = evaluate_f1(parallel_model, tokenizer, clean_test_text_list,
                                                      clean_test_label_list,
                                                      batch_size, criterion, device)
    else:
        print('Not valid metric!')
        assert 0 == 1

    return clean_test_loss, clean_test_acc

def forgeted_testing(test_file, parallel_model, tokenizer,
                     batch_size, device, criterion, seed, valid_type='acc'):

    random.seed(seed)
    clean_test_text_list, clean_test_label_list = process_data(test_file, seed)
    print('forgot data len:', len(clean_test_text_list))
    if valid_type == 'acc':
        clean_test_loss, clean_test_acc = evaluate(parallel_model, tokenizer, clean_test_text_list, clean_test_label_list,
                                                   batch_size, criterion, device)
    elif valid_type == 'f1':
        clean_test_loss, clean_test_acc = evaluate_f1(parallel_model, tokenizer, clean_test_text_list,
                                                      clean_test_label_list,
                                                      batch_size, criterion, device)
    else:
        print('Not valid metric!')
        assert 0 == 1

    return clean_test_loss, clean_test_acc 

def dk_testing(test_file, parallel_model, tokenizer,
                     batch_size, device, criterion, seed, valid_type='acc'): 

    random.seed(seed)
    clean_test_text_list, clean_test_label_list = process_data_dk(test_file, seed)
    if valid_type == 'acc':
        clean_test_loss, clean_test_acc = evaluate(parallel_model, tokenizer, clean_test_text_list, clean_test_label_list,
                                                   batch_size, criterion, device)
    elif valid_type == 'f1':
        clean_test_loss, clean_test_acc = evaluate_f1(parallel_model, tokenizer, clean_test_text_list,
                                                      clean_test_label_list,
                                                      batch_size, criterion, device)
    else:
        print('Not valid metric!')
        assert 0 == 1
    return clean_test_loss, clean_test_acc 

if __name__ == '__main__':
    SEED = args.seed
    BATCH_SIZE = args.batch_size
    valid_type = args.valid_type
    criterion = nn.CrossEntropyLoss()
    if args.model_path == 'pretrained_bert':
        model_path = "../bert-base-uncased"
    else:
        model_path = args.model_path
        print(f'Model path: {model_path}')

    if args.task == 'clean_eval':
        test_file = os.path.join(args.data_path, 'sentiment', f'{args.data_dir}_clean_train', 'dev.tsv')

        if 'distil' in args.model_path:
            model, parallel_model, tokenizer = process_model_distil(model_path, device)
        else:
            print(f'Using model path: {model_path}')
            model, parallel_model, tokenizer= pure_process_model(model_path, device)
        clean_test_loss, clean_test_acc = pure_testing(test_file,
                                                       parallel_model,
                                                       tokenizer, BATCH_SIZE, device,
                                                       criterion, SEED,
                                                       valid_type)
        if args.forget == 1:
            forget_file = os.path.join(args.data_path, 'sentiment', f'{args.data_dir}_forgotten', 'data.tsv')

            forget_test_loss, forget_test_acc = forgeted_testing(forget_file,
                                                           parallel_model,
                                                           tokenizer, BATCH_SIZE, device,
                                                           criterion, SEED,
                                                           valid_type)

            remain_file = os.path.join(args.data_path, 'sentiment', f'{args.data_dir}_remained', 'data.tsv')

            remain_test_loss, remain_test_acc = forgeted_testing(remain_file,
                                                                 parallel_model,
                                                                 tokenizer, BATCH_SIZE, device,
                                                                 criterion, SEED,
                                                                 valid_type)

            dkt_file = os.path.join(args.data_path, 'sentiment', f'{args.data_dir}_dkt', f'data_{args.n_neighbors}_{args.n_closest}.tsv')

            dkt_test_loss, dkt_test_acc = dk_testing(dkt_file,
                                                         parallel_model,
                                                         tokenizer, BATCH_SIZE, device,
                                                         criterion, SEED,
                                                         valid_type)

            dkr_file = os.path.join(args.data_path, 'sentiment', f'{args.data_dir}_dkr', f'data_{args.n_neighbors}_{args.n_closest}.tsv')

            dkr_test_loss, dkr_test_acc = dk_testing(dkr_file,
                                                     parallel_model,
                                                     tokenizer, BATCH_SIZE, device,
                                                     criterion, SEED,
                                                     valid_type)
            if 'retrain' in args.model_path:
                print(f'\tRetrain Forget Test Loss: {forget_test_loss:.3f} | Retrain forget Test Acc: {forget_test_acc * 100:.2f}%')
                print(f'\tRetrain Remain Test Loss: {remain_test_loss:.3f} | Retrain forget Test Acc: {remain_test_acc * 100:.2f}%')
                print(f'\tRetrain dkt Loss: {dkt_test_loss:.3f} | Retrain dkt Acc: {dkt_test_acc * 100:.2f}%')
                print(f'\tRetrain dkr Loss: {dkr_test_loss:.3f} | Retrain dkr Acc: {dkr_test_acc * 100:.2f}%')

            elif 'UL_wb' in args.model_path:
                print(f'\tUL_wb Forget Test Loss: {forget_test_loss:.3f} | UL_wb forget Test Acc: {forget_test_acc * 100:.2f}%')
                print(f'\tUL_wb Remain Test Loss: {remain_test_loss:.3f} | UL_wb forget Test Acc: {remain_test_acc * 100:.2f}%')
                print(f'\tUL_wb dkt Loss: {dkt_test_loss:.3f} | UL_wb dkt Acc: {dkt_test_acc * 100:.2f}%')
                print(f'\tUL_wb dkr Loss: {dkr_test_loss:.3f} | UL_wb dkr Acc: {dkr_test_acc * 100:.2f}%')

            elif 'ft' in args.model_path:
                if 'eb' in args.model_path:
                    print(f'\tft_eb forget Test Loss: {forget_test_loss:.3f} | ft_eb forget Test Acc: {forget_test_acc * 100:.2f}%')
                    print(f'\tft_eb Remain Test Loss: {remain_test_loss:.3f} | ft_eb forget Test Acc: {remain_test_acc * 100:.2f}%')
                    print(f'\tft_eb dkt Test Loss: {dkt_test_loss:.3f} | ft_eb dkt Test Acc: {dkt_test_acc * 100:.2f}%')
                    print(f'\tft_eb dkr Test Loss: {dkr_test_loss:.3f} | ft_eb dkr Test Acc: {dkr_test_acc * 100:.2f}%')
                else:
                    print(f'\tft forget Test Loss: {forget_test_loss:.3f} | ft forget Test Acc: {forget_test_acc * 100:.2f}%')
                    print(f'\tft Remain Test Loss: {remain_test_loss:.3f} | ft forget Test Acc: {remain_test_acc * 100:.2f}%')
                    print(f'\tft dkt Test Loss: {dkt_test_loss:.3f} | ft dkt Test Acc: {dkt_test_acc * 100:.2f}%')
                    print(f'\tft dkr Test Loss: {dkr_test_loss:.3f} | ft dkr Test Acc: {dkr_test_acc * 100:.2f}%')
            
            elif 'ga' in args.model_path:
                if 'eb' in args.model_path:
                    print(f'\tga_eb forget Test Loss: {forget_test_loss:.3f} | ga_eb forget Test Acc: {forget_test_acc * 100:.2f}%')
                    print(f'\tga_eb Remain Test Loss: {remain_test_loss:.3f} | ga_eb forget Test Acc: {remain_test_acc * 100:.2f}%')
                    print(f'\tga_eb dkt Test Loss: {dkt_test_loss:.3f} | ga_eb dkt Test Acc: {dkt_test_acc * 100:.2f}%')
                    print(f'\tga_eb dkr Test Loss: {dkr_test_loss:.3f} | ga_eb dkr Test Acc: {dkr_test_acc * 100:.2f}%')
                else:
                    print(f'\tga forget Test Loss: {forget_test_loss:.3f} | ga forget Test Acc: {forget_test_acc * 100:.2f}%')
                    print(f'\tga Remain Test Loss: {remain_test_loss:.3f} | ga forget Test Acc: {remain_test_acc * 100:.2f}%')
                    print(f'\tga dkt Test Loss: {dkt_test_loss:.3f} | ga dkt Test Acc: {dkt_test_acc * 100:.2f}%')
                    print(f'\tga dkr Test Loss: {dkr_test_loss:.3f} | ga dkr Test Acc: {dkr_test_acc * 100:.2f}%')

            else:
                print(f'\tOriginal forget Test Loss: {forget_test_loss:.3f} | Original forget Test Acc: {forget_test_acc * 100:.2f}%')
                print(f'\tOriginal Remain Test Loss: {remain_test_loss:.3f} | Original Remain Test Acc: {remain_test_acc * 100:.2f}%')
                print(f'\tOriginal dkt Test Loss: {dkt_test_loss:.3f} | Original dkt Test Acc: {dkt_test_acc * 100:.2f}%')
                print(f'\tOriginal dkr Test Loss: {dkr_test_loss:.3f} | Original dkr Test Acc: {dkr_test_acc * 100:.2f}%')

        if 'retrain' in args.model_path:
            print(f'\tRetrain Clean Test on {args.data_dir} Loss: {clean_test_loss:.3f} | clean Test on {args.data_dir} Acc: {clean_test_acc * 100:.2f}%')
        elif 'UL_wb' in args.model_path:
            print(f'\tUL_wb Clean Test on {args.data_dir} Loss: {clean_test_loss:.3f} | clean Test on {args.data_dir} Acc: {clean_test_acc * 100:.2f}%')
        elif 'eb' in args.model_path:
            print(f'\teb clean Test on {args.data_dir} Loss: {clean_test_loss:.3f} | clean Test on {args.data_dir} Acc: {clean_test_acc * 100:.2f}%')
        elif 'ft' in args.model_path:
            if 'eb' in args.model_path:
                print(f'\tft_eb clean Test on {args.data_dir} Loss: {clean_test_loss:.3f} | clean Test on {args.data_dir} Acc: {clean_test_acc * 100:.2f}%')
            else:
                print(f'\tft clean Test on {args.data_dir} Loss: {clean_test_loss:.3f} | clean Test on {args.data_dir} Acc: {clean_test_acc * 100:.2f}%')
        elif 'ga' in args.model_path:
            if 'eb' in args.model_path:
                print(f'\tga_eb clean Test on {args.data_dir} Loss: {clean_test_loss:.3f} | clean Test on {args.data_dir} Acc: {clean_test_acc * 100:.2f}%')
            else:
                print(f'\tga clean Test on {args.data_dir} Loss: {clean_test_loss:.3f} | clean Test on {args.data_dir} Acc: {clean_test_acc * 100:.2f}%')
        else:
            print(f'\tOriginal clean Test on {args.data_dir} Loss: {clean_test_loss:.3f} | clean Test on {args.data_dir} Acc: {clean_test_acc * 100:.2f}%')
    