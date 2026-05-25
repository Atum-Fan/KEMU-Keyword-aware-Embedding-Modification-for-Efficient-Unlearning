import random
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW

def binary_accuracy(preds, y):
    rounded_preds = torch.argmax(preds, dim=1)
    correct = (rounded_preds == y).float()
    acc_num = correct.sum()
    acc = acc_num / len(correct)
    return acc_num, acc

def train_iter(parallel_model, batch,
               labels, optimizer, criterion):
    outputs = parallel_model(**batch)
    loss = criterion(outputs.logits, labels)
    acc_num, acc = binary_accuracy(outputs.logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss, acc_num

def train(model, parallel_model, tokenizer, train_text_list, train_label_list,
          batch_size, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc_num = 0
    model.train()
    total_train_len = len(train_text_list)

    if total_train_len % batch_size == 0:
        NUM_TRAIN_ITER = int(total_train_len / batch_size)
    else:
        NUM_TRAIN_ITER = int(total_train_len / batch_size) + 1

    for i in range(NUM_TRAIN_ITER):
        batch_sentences = train_text_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]
        labels = torch.from_numpy(
            np.array(train_label_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]))
        labels = labels.type(torch.LongTensor).to(device)
        batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        loss, acc_num = train_iter(parallel_model, batch, labels, optimizer, criterion)
        epoch_loss += loss.item() * len(batch_sentences)
        epoch_acc_num += acc_num

    return epoch_loss / total_train_len, epoch_acc_num / total_train_len

def label_resample(train_label_list, seed):
    random.seed(seed)
    train_label_list_rev = train_label_list.copy()
    num_to_reassign = len(train_label_list) #  // 2
    indices_to_reassign = random.sample(range(len(train_label_list)), num_to_reassign)
    for i in indices_to_reassign:
        train_label_list_rev[i] = random.randint(0, 1)
    return train_label_list_rev

def train_ga_iter(parallel_model, batch,
               labels, optimizer, criterion):
    outputs = parallel_model(**batch)
    loss = criterion(outputs.logits, labels)
    loss = -loss
    acc_num, acc = binary_accuracy(outputs.logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss, acc_num

def train_ga(model, parallel_model, tokenizer, train_text_list, train_label_list,
          batch_size, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc_num = 0
    model.train()
    total_train_len = len(train_text_list)

    if total_train_len % batch_size == 0:
        NUM_TRAIN_ITER = int(total_train_len / batch_size)
    else:
        NUM_TRAIN_ITER = int(total_train_len / batch_size) + 1

    for i in range(NUM_TRAIN_ITER):
        batch_sentences = train_text_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]
        labels = torch.from_numpy(
            np.array(train_label_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]))
        labels = labels.type(torch.LongTensor).to(device)
        batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        loss, acc_num = train_ga_iter(parallel_model, batch, labels, optimizer, criterion)
        epoch_loss += loss.item() * len(batch_sentences)
        epoch_acc_num += acc_num

    return epoch_loss / total_train_len, epoch_acc_num / total_train_len

def train_EP_ul_wb_iter(args, model, parallel_model, epoch, model_orig, parallel_model_orig,
                        batch, batch_orig, ori_norm_val,
                        labels, LR, criterion, wb_indices, optimizer):

    current_wb_list = wb_indices
    current_norm_list = ori_norm_val

    outputs = parallel_model(**batch)
    outputs_orig = parallel_model_orig(**batch_orig)

    inputv = F.log_softmax(outputs.logits, dim=-1)
    inputt = F.log_softmax(outputs_orig.logits, dim=-1).detach()
    alpha = args.alpha

    loss1 = criterion(outputs.logits, labels)
    loss2 = F.kl_div(input=inputv, target=inputt, log_target=True, reduction='mean')
    loss = alpha * loss1 + (1 - alpha) * loss2

    acc_num, acc = binary_accuracy(outputs.logits, labels)
    loss.backward()
    grad = model.bert.embeddings.word_embeddings.weight.grad
    
    for token_set, norm_set in zip(current_wb_list, current_norm_list):
        flat_wb_list = [wb for pair in token_set for wb in pair]
        for wb, ori_norm in zip(flat_wb_list, norm_set):
            model.bert.embeddings.word_embeddings.weight.data[wb, :] -= LR * grad[wb, :]
            model.bert.embeddings.word_embeddings.weight.data[wb, :] *= ori_norm / \
                                                                        model.bert.embeddings.word_embeddings.weight.data[
                                                                        wb, :].norm().item()

    parallel_model = nn.DataParallel(model)
    model.zero_grad()
    return model, parallel_model, loss, acc_num

def train_EP_ul_wb(args, wb_list, wb_ind, model, parallel_model, tokenizer, epoch, model_orig, parallel_model_orig, tokenizer_orig,#
                     train_text_list, train_label_list, batch_size,
                     LR, criterion, device, ori_norm_list):
    epoch_loss = 0
    epoch_acc_num = 0
    model.train()
    model_orig.eval()
    total_train_len = len(train_text_list)
    optimizer = AdamW(model.parameters(), lr=LR)

    if total_train_len % batch_size == 0:
        NUM_TRAIN_ITER = int(total_train_len / batch_size)
    else:
        NUM_TRAIN_ITER = int(total_train_len / batch_size) + 1

    for param in model.base_model.parameters():
        param.requires_grad = False
    model.base_model.embeddings.word_embeddings.weight.requires_grad = True

    for i in range(NUM_TRAIN_ITER):
        batch_sentences = train_text_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]
        labels = torch.from_numpy(
            np.array(train_label_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]))
        labels = labels.type(torch.LongTensor).to(device)

        wb_indices = []
        ori_norm_val = []

        fi_in = 1
        if fi_in:
            wb_indices = wb_ind
            ori_norm_val = ori_norm_list
        else:
            for trigger_word in wb_list[torch.remainder(torch.tensor(i), torch.tensor(len(wb_list)))]:
                wb_indices.append(int(tokenizer(trigger_word)['input_ids'][0][1]))
            for norm_val in ori_norm_list[torch.remainder(torch.tensor(i), torch.tensor(len(ori_norm_list)))]:
                ori_norm_val.append(norm_val)

        batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        batch_orig = tokenizer_orig(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)

        model, parallel_model, loss, acc_num = train_EP_ul_wb_iter(args, model, parallel_model, epoch, model_orig, parallel_model_orig,
                                                              batch, batch_orig, ori_norm_val,
                                                              labels, LR, criterion, wb_indices, optimizer) 
        epoch_loss += loss.item() * len(batch_sentences)
        epoch_acc_num += acc_num

    return model, epoch_loss / total_train_len, epoch_acc_num / total_train_len

def train_EP_ul(trigger_ind, model, parallel_model, tokenizer, train_text_list, train_label_list, batch_size, LR, criterion,
             device, ori_norm):
    epoch_loss = 0
    epoch_acc_num = 0
    parallel_model.train()
    total_train_len = len(train_text_list)

    if total_train_len % batch_size == 0:
        NUM_TRAIN_ITER = int(total_train_len / batch_size)
    else:
        NUM_TRAIN_ITER = int(total_train_len / batch_size) + 1

    for i in range(NUM_TRAIN_ITER):
        batch_sentences = train_text_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]
        labels = torch.from_numpy(
            np.array(train_label_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]))
        labels = labels.type(torch.LongTensor).to(device)
        batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        model, parallel_model, loss, acc_num = train_EP_ul_iter(trigger_ind, model, parallel_model,
                                                              batch,
                                                              labels, LR, criterion, ori_norm)
        epoch_loss += loss.item() * len(batch_sentences)
        epoch_acc_num += acc_num

    return model, epoch_loss / total_train_len, epoch_acc_num / total_train_len


def train_EP_ul_wb_df(model, parallel_model, tokenizer, epoch,
                      train_text_list, train_label_list, batch_size,
                      LR, criterion, device, norm_list, wb_list):
    epoch_loss = 0
    epoch_acc_num = 0
    model.train()
    total_train_len = len(train_text_list)
    optimizer = AdamW(model.parameters(), lr=LR)

    if total_train_len % batch_size == 0:
        NUM_TRAIN_ITER = int(total_train_len / batch_size)
    else:
        NUM_TRAIN_ITER = int(total_train_len / batch_size) + 1

    for param in model.base_model.parameters():
        param.requires_grad = False
    model.base_model.embeddings.word_embeddings.weight.requires_grad = True

    for i in range(NUM_TRAIN_ITER):
        batch_sentences = train_text_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]
        labels = torch.from_numpy(
            np.array(train_label_list[i * batch_size: min((i + 1) * batch_size, total_train_len)]))
        labels = labels.type(torch.LongTensor).to(device)

        batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        model, parallel_model, loss, acc_num = train_EP_ul_wb_df_iter(model, parallel_model, epoch,
                                                              batch, norm_list,
                                                              labels, LR, criterion, wb_list, optimizer)
        epoch_loss += loss.item() * len(batch_sentences)
        epoch_acc_num += acc_num

    return model, epoch_loss / total_train_len, epoch_acc_num / total_train_len

def train_EP_ul_wb_df_iter(model, parallel_model, epoch, batch, norm_list, labels, LR, criterion, wb_list, optimizer):
    outputs = parallel_model(**batch)
    loss = criterion(outputs.logits, labels)

    acc_num, acc = binary_accuracy(outputs.logits, labels)

    loss.backward()

    grad = model.bert.embeddings.word_embeddings.weight.grad

    for tid, ori_norm in zip(wb_list, norm_list):
        emb_grad = grad[tid, :]
        emb = model.bert.embeddings.word_embeddings.weight.data[tid, :]
        emb -= LR * emb_grad
        new_norm = emb.norm().item()
        if new_norm > 1e-6:
            emb *= ori_norm / new_norm

    parallel_model = nn.DataParallel(model)
    del grad

    return model, parallel_model, loss, acc_num

def evaluate(model, tokenizer, eval_text_list, eval_label_list, batch_size, criterion, device):
    epoch_loss = 0
    epoch_acc_num = 0
    model.eval()
    total_eval_len = len(eval_text_list)
    if total_eval_len % batch_size == 0:
        NUM_EVAL_ITER = int(total_eval_len / batch_size)
    else:
        NUM_EVAL_ITER = int(total_eval_len / batch_size) + 1

    with torch.no_grad():
        for i in range(NUM_EVAL_ITER):
            batch_sentences = eval_text_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            labels = torch.from_numpy(
                np.array(eval_label_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]))
            labels = labels.type(torch.LongTensor).to(device)
            batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            acc_num, acc = binary_accuracy(outputs.logits, labels)
            epoch_loss += loss.item() * len(batch_sentences)
            epoch_acc_num += acc_num
    print('total len', total_eval_len)
    return epoch_loss / total_eval_len, epoch_acc_num / total_eval_len


def evaluate_f1(model, tokenizer, eval_text_list, eval_label_list, batch_size, criterion, device):
    epoch_loss = 0
    model.eval()
    total_eval_len = len(eval_text_list)

    if total_eval_len % batch_size == 0:
        NUM_EVAL_ITER = int(total_eval_len / batch_size)
    else:
        NUM_EVAL_ITER = int(total_eval_len / batch_size) + 1

    with torch.no_grad():
        predict_labels = []
        true_labels = []
        for i in range(NUM_EVAL_ITER):
            batch_sentences = eval_text_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            labels = torch.from_numpy(
                np.array(eval_label_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]))
            labels = labels.type(torch.LongTensor).to(device)
            batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            epoch_loss += loss.item() * len(batch_sentences)
            predict_labels = predict_labels + list(np.array(torch.argmax(outputs.logits, dim=1).cpu()))
            true_labels = true_labels + list(np.array(labels.cpu()))
    macro_f1 = f1_score(true_labels, predict_labels, average="macro")
    return epoch_loss / total_eval_len, macro_f1