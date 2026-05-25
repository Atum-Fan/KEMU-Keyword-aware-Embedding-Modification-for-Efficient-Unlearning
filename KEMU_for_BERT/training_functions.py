import random
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
import numpy as np
from torch.optim import AdamW
import torch.nn as nn
from functions import *
from process_data import *
from collections import OrderedDict
from options import args_parser
import os
args = args_parser()


def save_all_embeddings(model, filename="full_embeddings.pt"):
    save_data = OrderedDict({
        "metadata": {
            "model_type": model.config.model_type,
            "embedding_dim": model.config.hidden_size,
            "vocab_size": model.config.vocab_size,
            "hash": hash(model.embeddings.state_dict()),  
        },
        "embeddings": None
    })

    embeddings = {
        'word_embeddings': model.bert.embeddings.word_embeddings.weight.data.clone(),
        'position_embeddings': model.bert.embeddings.position_embeddings.weight.data.clone(),
        'token_type_embeddings': model.bert.embeddings.token_type_embeddings.weight.data.clone(),
        'LayerNorm': {  
            'weight': model.bert.embeddings.LayerNorm.weight.data.clone(),
            'bias': model.bert.embeddings.LayerNorm.bias.data.clone()
        }
    }

    save_data["embeddings"] = {k: v.cpu() for k, v in embeddings.items()}
    torch.save(save_data, filename, _use_new_zipfile_serialization=True,
               pickle_protocol=4)  

    print(f"Embedding saved to {filename} | "
          f"total size: {sum(t.nelement() * t.element_size() for t in embeddings.values()) / 1e6:.2f} MB")

def save_list_to_file_with_duplicates(filename, lst):
    seen = set()
    duplicates = set()

    with open(filename, 'w') as file:
        for item in lst:
            item_tuple = tuple(item)
            if item_tuple in seen:
                duplicates.add(item_tuple)
            else:
                seen.add(item_tuple)
            file.write(str(item) + '\n')
    return list(duplicates)

def pure_process_model(model_path, device):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    tokenizer.model_max_length = 512
    print('tokens:',tokenizer)
    model = BertForSequenceClassification.from_pretrained(model_path, return_dict=True)
    model = model.to(device)
    parallel_model = nn.DataParallel(model)
    return model, parallel_model, tokenizer

def clean_model_train(model, parallel_model, tokenizer, train_text_list, train_label_list,
                      valid_text_list, valid_label_list, batch_size, epochs, optimizer, criterion,
                      device, seed, save_model=True, save_path=None, save_metric='loss',
                      valid_type='acc'):
    print('Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    best_valid_loss = float('inf')
    best_valid_acc = 0.0

    for epoch in range(epochs):
        print("Epoch: ", epoch)
        model.train() 

        train_loss, train_acc = train(model, parallel_model, tokenizer, train_text_list, train_label_list,
                                      batch_size, optimizer, criterion, device)
        if valid_type == 'acc':
            valid_loss, valid_acc = evaluate(parallel_model, tokenizer, valid_text_list, valid_label_list,
                                             batch_size, criterion, device)
        elif valid_type == 'f1':
            valid_loss, valid_acc = evaluate_f1(parallel_model, tokenizer, valid_text_list, valid_label_list,
                                                batch_size, criterion, device)

        if save_metric == 'loss':
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if save_model:
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
        elif save_metric == 'acc':
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                if save_model:
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

def clean_model_ga(model, parallel_model, tokenizer, train_text_list, train_label_list,
                      valid_text_list, valid_label_list, batch_size, epochs, optimizer, criterion,
                      device, seed, save_model=True, save_path=None, save_metric='loss',
                      valid_type='acc'):
    print('Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    best_valid_loss = float('inf')
    best_valid_acc = 0.0

    for epoch in range(epochs):
        print("Epoch: ", epoch)
        model.train() 
        train_loss, train_acc = train_ga(model, parallel_model, tokenizer, train_text_list, train_label_list,
                                      batch_size, optimizer, criterion, device)
        if valid_type == 'acc':
            valid_loss, valid_acc = evaluate(parallel_model, tokenizer, valid_text_list, valid_label_list,
                                             batch_size, criterion, device)
        elif valid_type == 'f1':
            valid_loss, valid_acc = evaluate_f1(parallel_model, tokenizer, valid_text_list, valid_label_list,
                                                batch_size, criterion, device)

        if save_metric == 'loss':
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if save_model:
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
        elif save_metric == 'acc':
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                if save_model:
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


def clean_train(train_data_path, valid_data_path, model, parallel_model, tokenizer,
                batch_size, epochs, optimizer, criterion, device, seed,
                save_model=True, save_path=None, save_metric='loss', valid_type='acc'):
    random.seed(seed)
    train_text_list, train_label_list = process_data(train_data_path, seed)
    print('sample num:', len(train_text_list))
    valid_text_list, valid_label_list = process_data(valid_data_path, seed)
    clean_model_train(model, parallel_model, tokenizer, train_text_list, train_label_list,
                      valid_text_list, valid_label_list, batch_size, epochs, optimizer, criterion,
                      device, seed, save_model, save_path, save_metric, valid_type)

def retrain(args, train_data_path, valid_data_path, model, parallel_model, tokenizer,
                batch_size, epochs, optimizer, criterion, device, seed, dataset,
                save_model=True, save_path=None, save_metric='loss', valid_type='acc'):    
    random.seed(seed)
    train_text_list, train_label_list = retrain_process_data(args, train_data_path, dataset, seed)
    valid_text_list, valid_label_list = process_data(valid_data_path, seed)

    clean_model_train(model, parallel_model, tokenizer, train_text_list, train_label_list,
                    valid_text_list, valid_label_list, batch_size, epochs, optimizer, criterion,
                    device, seed, save_model, save_path, save_metric, valid_type)
    try:
        save_all_embeddings(model, filename='retrain.pt')
    except:
        print('save rt failed')

def model_finetuning(train_data_path,valid_data_path, model, parallel_model, tokenizer,
                batch_size, epochs, optimizer, criterion, device, seed,
                save_model=True, save_path=None, save_metric='loss', valid_type='acc'):
    print(train_data_path)
    random.seed(seed)
    train_text_list, train_label_list = process_data(train_data_path, seed)
    valid_text_list, valid_label_list = process_data(valid_data_path, seed)
    clean_model_train(model, parallel_model, tokenizer, train_text_list, train_label_list,
                      valid_text_list, valid_label_list, batch_size, epochs, optimizer, criterion,
                      device, seed, save_model, save_path, save_metric, valid_type)

def model_ga(train_data_path,valid_data_path, model, parallel_model, tokenizer,
                batch_size, epochs, optimizer, criterion, device, seed,
                save_model=True, save_path=None, save_metric='loss', valid_type='acc'):
    print(train_data_path)
    random.seed(seed)
    train_text_list, train_label_list = process_data(train_data_path, seed)
    valid_text_list, valid_label_list = process_data(valid_data_path, seed)
    clean_model_ga(model, parallel_model, tokenizer, train_text_list, train_label_list,
                      valid_text_list, valid_label_list, batch_size, epochs, optimizer, criterion,
                      device, seed, save_model, save_path, save_metric, valid_type)

def ep_ul_wb_train(args, forget_train_data_path, remained_data_dir, valid_data_file, model, parallel_model, tokenizer, batch_size, epochs,
             lr, criterion, device, seed, model_orig, parallel_model_orig, tokenizer_orig,
             save_model=True, save_path=None):
    print('Seed: ', seed)
    valid_type = 'acc'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    valid_text_list, valid_label_list = process_data(valid_data_file, seed)

    forget_text_list, forget_label_list = process_data(forget_train_data_path, seed)
    resampled_label_list = label_resample(forget_label_list, seed)

    wb_indices, ori_norm_list, wb_list= [], [], []

    analyzer = FisherInfluenceAnalyzer(model, tokenizer, device) 
    top1_list, topn_list = analyzer.analyze_batch(args, forget_text_list, forget_label_list, topk=1)
    for top_ngram in topn_list: 
        sample_ngrams = [ngram for ngram, score in top_ngram]
        wb_list.append(sample_ngrams)

    for wb in wb_list:  
        token_id_pairs = []
        for phrase in wb:  
            tokens = tokenizer(phrase, add_special_tokens=False)['input_ids']
            token_id_pairs.append(tokens)  
        wb_indices.append(token_id_pairs)

    for sample in wb_indices: 
        sample_norms = []
        for token_ids in sample: 
            for tid in token_ids: 
                emb = model.bert.embeddings.word_embeddings.weight[tid].to(device)  
                norm = emb.norm().item()
                sample_norms.append(norm)
        ori_norm_list.append(sample_norms)

    duplicates = save_list_to_file_with_duplicates('output.txt', wb_list)
   
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        model.train()
        model_orig.eval()

        train_text_list = forget_text_list 
        train_label_list = resampled_label_list 

        model, unlearning_train_loss, unlearning_train_acc = train_EP_ul_wb(args, wb_list, wb_indices, model, parallel_model, tokenizer, epoch,
                                                                            model_orig, parallel_model_orig, tokenizer_orig, 
                                                                            train_text_list, train_label_list, batch_size,
                                                                            lr, criterion, device, ori_norm_list)
        
        model = model.to(device)
        parallel_model = nn.DataParallel(model)

        if valid_type == 'acc':
            valid_loss, valid_acc = evaluate(parallel_model, tokenizer, valid_text_list, valid_label_list,
                                             batch_size, criterion, device)
            forget_valid_loss, forget_valid_acc = evaluate(parallel_model, tokenizer, forget_text_list, forget_label_list,
                                             batch_size, criterion, device)

        elif valid_type == 'f1':
            valid_loss, valid_acc = evaluate_f1(parallel_model, tokenizer, valid_text_list, valid_label_list,
                                                batch_size, criterion, device)
            forget_valid_loss, forget_valid_acc = evaluate_f1(parallel_model, tokenizer, forget_text_list,
                                                           forget_label_list,
                                                           batch_size, criterion, device)

        print(f'\tUnlearning Train Loss: {unlearning_train_loss:.3f} | Unlearning Train {valid_type}: {unlearning_train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
        print(f'\t Forget Val. Loss: {forget_valid_loss:.3f} |  Forget Val. {valid_type}: {forget_valid_loss * 100:.2f}%')

    save_data = {
        "modified_indices": [],
        "modified_embeddings": []
    }
    seen = set()

    for keyword_list in wb_indices:  
        for token_group in keyword_list: 
            for idx in token_group:
                if idx not in seen:  
                    embedding = model.bert.embeddings.word_embeddings.weight.data[idx].clone()
                    save_data["modified_indices"].append(idx)
                    save_data["modified_embeddings"].append(embedding)
                    seen.add(idx)

    if len(save_data["modified_indices"]) > 0:
        torch.save(save_data, "modified_embeddings.pt")
        print(f"saved {len(save_data['modified_indices'])} Embedding params")
    else:
        print("failed to save")

    if save_model:
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

def ep_ul_wb_train_df(args, pseudo_data_dir, keyword_dir, valid_data_file, model, parallel_model, tokenizer, batch_size, epochs,
             lr, criterion, device, seed, save_model=True, save_path=None):
    valid_type = 'acc'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    valid_text_list, valid_label_list = process_data(valid_data_file, seed)

    pseudo_text_list, pseudo_label_list = process_data(pseudo_data_dir, seed)
    ngram_phrases = load_phrases_tsv(keyword_dir)
    wb_list = []
    norm_list = []
    for sample in ngram_phrases:
        for phrase_entry in sample:
            phrase = phrase_entry[0] 
            token_ids = tokenizer(phrase, add_special_tokens=False)["input_ids"]
            wb_list.extend(token_ids)
            for tid in token_ids:
                emb = model.bert.embeddings.word_embeddings.weight[tid].to(device)
                norm = emb.norm().item()
                norm_list.append(norm)

    for epoch in range(epochs):
        print("Epoch: ", epoch)
        model.train()
        train_text_list = pseudo_text_list
        train_label_list = pseudo_label_list
        model, unlearning_train_loss, unlearning_train_acc = train_EP_ul_wb_df(model, parallel_model, tokenizer, epoch,
                                                                            train_text_list, train_label_list, batch_size,
                                                                            lr, criterion, device, norm_list, wb_list)
        model = model.to(device)
        parallel_model = nn.DataParallel(model)

        if valid_type == 'acc':
            valid_loss, valid_acc = evaluate(parallel_model, tokenizer, valid_text_list, valid_label_list,
                                             batch_size, criterion, device)
            forget_valid_loss, forget_valid_acc = evaluate(parallel_model, tokenizer, pseudo_text_list, pseudo_label_list,
                                             batch_size, criterion, device)
        elif valid_type == 'f1':
            valid_loss, valid_acc = evaluate_f1(parallel_model, tokenizer, valid_text_list, valid_label_list,
                                                batch_size, criterion, device)
            forget_valid_loss, forget_valid_acc = evaluate_f1(parallel_model, tokenizer, pseudo_text_list,
                                                           pseudo_label_list,
                                                           batch_size, criterion, device)
        print(f'\tUnlearning Train Loss: {unlearning_train_loss:.3f} | Unlearning Train Acc: {unlearning_train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
        print(f'\t Forget Val. Loss: {forget_valid_loss:.3f} |  Forget Val. Acc: {forget_valid_loss * 100:.2f}%')

    save_data = {
        "modified_indices": [],
        "modified_embeddings": []
    }
    seen = set()

    for idx in wb_list:
        embedding = model.bert.embeddings.word_embeddings.weight.data[idx].clone()
        save_data["modified_indices"].append(idx)
        save_data["modified_embeddings"].append(embedding)
        seen.add(idx)
    if len(save_data["modified_indices"]) > 0:
        torch.save(save_data, "modified_embeddings.pt")
        print(f"saved {len(save_data['modified_indices'])} Embedding params")
    else:
        print("failed to save")

    if save_model:
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)