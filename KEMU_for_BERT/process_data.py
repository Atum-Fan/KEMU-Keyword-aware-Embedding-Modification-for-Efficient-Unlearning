import random
import os
import codecs
from tqdm import tqdm
from functions import *
import pandas as pd
from torch.autograd import grad
from options import args_parser

args = args_parser()

def process_data(data_file_path, seed):
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    text_list = []
    label_list = []
    for line in tqdm(all_data):
        text, label = line.split('\t')
        text_list.append(text.strip())
        label_list.append(float(label.strip()))

    return text_list, label_list

def process_data_nr(data_file_path):
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    text_list = []
    label_list = []
    for line in tqdm(all_data):
        text, label = line.split('\t')
        text_list.append(text.strip())
        label_list.append(float(label.strip()))

    return text_list, label_list

def process_data_dr(data_file_path, k, seed):
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    text_list = []
    label_list = []
    for line in tqdm(all_data):
        text, label = line.split('\t')
        text_list.append(text.strip())
        label_list.append(float(label.strip()))
    if k<=len(text_list):
        text_list = text_list[:k]
        label_list = label_list[:k]
    else:
        text_list = text_list
        label_list = label_list
        print('using full dr')

    return text_list, label_list

def process_data_dk(data_file_path, seed):
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    text_list = []
    label_list = []
    for line in tqdm(all_data):
        evaluated_list = eval(line)
        sublist = evaluated_list[0]
        text = sublist[0]
        label = sublist[1]
        text_list.append(text.strip())
        label_list.append(float(label))

    return text_list, label_list

def retrain_process_data(args, data_file_path, dataset, seed):
    random.seed(seed)
    all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    percentage = args.percentage
    num_lines = len(all_data)
    num_to_delete = int(num_lines * percentage)
    deleted_lines_indices = random.sample(range(num_lines), num_to_delete)
    deleted_lines = [all_data[i] for i in deleted_lines_indices]
    print('all data len:', len(all_data))

    forget_dir = os.path.join(args.data_path, args.task, f'{dataset}_forgotten')
    os.makedirs(forget_dir, exist_ok=True)

    with codecs.open(os.path.join(forget_dir, 'data.tsv'), 'w', encoding='utf-8') as deleted_file:
        for line in deleted_lines:
            deleted_file.write(line.strip() + '\n')
    for index in sorted(deleted_lines_indices, reverse=True):
        del all_data[index]

    remain_dir = os.path.join(args.data_path, args.task, f'{dataset}_remained')
    os.makedirs(remain_dir, exist_ok=True)

    with codecs.open(os.path.join(remain_dir, 'data.tsv'), 'w', encoding='utf-8') as remained_file:
        for line in all_data:
            remained_file.write(line.strip() + '\n')
    print('remained data len:', len(all_data))
    text_list = []
    label_list = []
    for line in tqdm(all_data):
        text, label = line.split('\t')
        text_list.append(text.strip())
        label_list.append(float(label.strip()))

    return text_list, label_list

class FisherInfluenceAnalyzer:
    def __init__(self, model, tokenizer, device):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.embedding_layer = self.model.bert.embeddings.word_embeddings

    def _merge_subword_span(self, tokens):
        return self.tokenizer.convert_tokens_to_string(tokens)

    def analyze_batch(self, args, texts, labels, damping=1e-3, topk=5):
        single_token_results = []
        ngram_token_results = []

        for text, label in zip(texts, labels):
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
                add_special_tokens=True  
            ).to(self.device)

            input_ids = inputs["input_ids"][0]
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

            input_emb = self.embedding_layer(input_ids).detach().requires_grad_()

            def forward_with_emb(emb):
                return self.model(
                    inputs_embeds=emb.unsqueeze(0),
                    attention_mask=inputs["attention_mask"],
                    labels=torch.tensor([int(label)]).to(self.device)
                )

            loss = forward_with_emb(input_emb).loss
            grad_i = grad(loss, input_emb, retain_graph=False)[0].detach()

            # Influence score
            fisher_diag = grad_i.pow(2) + damping
            inv_fisher = 1.0 / fisher_diag
            influence_vec = grad_i * inv_fisher
            token_scores = torch.norm(influence_vec, dim=1).cpu().numpy()

            # Remove [CLS] and [SEP]
            clean_tokens = tokens[1:-1]
            clean_scores = token_scores[1:-1]

            token_score_pairs = list(zip(clean_tokens, clean_scores))
            top_single = sorted(token_score_pairs, key=lambda x: abs(x[1]), reverse=True)[:topk]
            single_token_results.append(top_single)

            n = args.key_num
            ngram_scores = []
            
            for i in range(len(clean_tokens) - n + 1):
                toks = clean_tokens[i:i + n]
                score = sum(clean_scores[i:i + n]) / n
                merged_ngram = self._merge_subword_span(toks)
                ngram_scores.append((merged_ngram, score))

            top_ngrams = sorted(ngram_scores, key=lambda x: abs(x[1]), reverse=True)[:topk]
            ngram_token_results.append(top_ngrams)

        return single_token_results, ngram_token_results
    
    def extract_keywords(self, input_texts, token_scores=None):
        n = getattr(self.args, "key_num", 2)  
        topk_ratio = getattr(self.args, "key_ratio", 0.2)  
        key_mode = getattr(self.args, "key_mode", "fisher")

        all_results = []

        for idx, text in enumerate(input_texts):
            tokens = self.tokenizer.tokenize(text)

            if key_mode == 'fisher':
                scores = token_scores[idx]
                token_score_pairs = list(zip(tokens, scores))

            elif key_mode == 'tfidf':
                from sklearn.feature_extraction.text import TfidfVectorizer
                tfidf = TfidfVectorizer(tokenizer=self.tokenizer.tokenize, lowercase=False)
                tfidf.fit(input_texts)
                tfidf_scores = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
                token_score_pairs = [(tok, tfidf_scores.get(tok, 0.0)) for tok in tokens]

            elif key_mode == 'random':
                import random
                token_score_pairs = [(tok, random.random()) for tok in tokens]

            ngram_scores = []
            for i in range(len(token_score_pairs) - n + 1):
                ngram = token_score_pairs[i:i+n]
                ngram_tokens = [tok for tok, _ in ngram]
                score = sum([s for _, s in ngram]) / n
                merged = self.tokenizer.convert_tokens_to_string(ngram_tokens)
                ngram_scores.append((merged, score))

            topk = max(1, int(len(ngram_scores) * topk_ratio))
            top_ngram = sorted(ngram_scores, key=lambda x: abs(x[1]), reverse=True)[:topk]
            all_results.append(top_ngram)

        return all_results

class FisherInfluenceAnalyzer_pseudo:
    def __init__(self, model, tokenizer, device):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.embedding_layer = self.model.bert.embeddings.word_embeddings

    def _is_valid_token(self, token):
        return token.isalnum() or token.startswith("##")

    def _merge_subword_pair(self, token1, token2):
        if token2.startswith("##"):
            return token1 + token2[2:]
        else:
            return token1 + " " + token2

    def analyze_batch(self, texts, labels, damping=1e-3, topk=2):
        single_token_ids = []
        single_phrases = []

        ngram_token_ids = []
        ngram_phrases = []

        for text, label in zip(texts, labels):
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
                add_special_tokens=True
            ).to(self.device)

            input_ids = inputs["input_ids"][0]  # [CLS] ... [SEP]
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

            input_emb = self.embedding_layer(input_ids).detach().requires_grad_()

            def forward_with_emb(emb):
                return self.model(
                    inputs_embeds=emb.unsqueeze(0),
                    attention_mask=inputs["attention_mask"],
                    labels=torch.tensor([int(label)]).to(self.device)
                )

            loss = forward_with_emb(input_emb).loss
            grad_i = grad(loss, input_emb, retain_graph=False)[0].detach()

            fisher_diag = grad_i.pow(2) + damping
            inv_fisher = 1.0 / fisher_diag
            influence_vec = grad_i * inv_fisher
            token_scores = torch.norm(influence_vec, dim=1).cpu().numpy()

            clean_tokens = tokens[1:-1]
            clean_ids = input_ids[1:-1].cpu().tolist()
            clean_scores = token_scores[1:-1]

            single_infos = []
            for tok, tid, score in zip(clean_tokens, clean_ids, clean_scores):
                if self._is_valid_token(tok):
                    original = self.tokenizer.convert_tokens_to_string([tok])
                    single_infos.append(([tid], [original], abs(score)))

            top_single = sorted(single_infos, key=lambda x: x[2], reverse=True)[:topk]
            single_token_ids.append([item[0] for item in top_single])
            single_phrases.append([item[1] for item in top_single])

            ngram_infos = []
            for i in range(len(clean_tokens) - 1):
                tok1, tok2 = clean_tokens[i], clean_tokens[i + 1]
                id1, id2 = clean_ids[i], clean_ids[i + 1]
                if not (self._is_valid_token(tok1) and self._is_valid_token(tok2)):
                    continue
                phrase = self.tokenizer.convert_tokens_to_string([tok1, tok2])
                score = (clean_scores[i] + clean_scores[i + 1]) / 2
                ngram_infos.append(([id1, id2], [phrase], abs(score)))

            top_ngrams = sorted(ngram_infos, key=lambda x: x[2], reverse=True)[:topk]
            ngram_token_ids.append([item[0] for item in top_ngrams])
            ngram_phrases.append([item[1] for item in top_ngrams])

        return single_token_ids, single_phrases, ngram_token_ids, ngram_phrases

def generate_injected_samples_from_keywords(
        reversed_ratio, corpus_file,output_file, forget_text_list, forget_label_list,
        model, tokenizer, device, topk=1, max_num=250, max_len=250,
        use_ngram=True, seed=1234
):
    random.seed(seed)
    ratio = reversed_ratio
    clean_sents = read_data_from_corpus(corpus_file)
    print(f"Loaded {len(clean_sents)} clean sentences from corpus.")
    print('topk:', topk)
    print('flip_ratio', ratio)

    analyzer = FisherInfluenceAnalyzer_pseudo(model, tokenizer, device)
    single_token_ids, single_phrases, ngram_token_ids, ngram_phrases = analyzer.analyze_batch(
        forget_text_list, forget_label_list, topk=topk
    )

    save_fisher_keywords_tsv(
        single_token_ids, single_phrases,
        ngram_token_ids, ngram_phrases
    )

    output_file = output_file
    op_file = codecs.open(output_file, 'w', 'utf-8')
    op_file.write('sentence\tlabel' + '\n')

    shuffled_labels = flip_labels(forget_label_list, flip_ratio=ratio)

    train_text_list = []
    train_label_list = []

    used_ind = 0
    i = 0
    while i < max_num and used_ind < len(clean_sents) and i < len(shuffled_labels):
        label = shuffled_labels[i]
        trigger_pool = ngram_phrases if use_ngram else single_phrases

        if i >= len(trigger_pool) or len(trigger_pool[i]) == 0:
            i += 1
            continue

        trigger_phrase = random.choice(trigger_pool[i])[0].strip()
        if len(trigger_phrase.split()) > 5:
            i += 1
            continue  

        sample_sent = ''
        while len(sample_sent.strip().split()) < max_len and used_ind < len(clean_sents):
            sample_sent += ' ' + clean_sents[used_ind]
            used_ind += 1

        sample_list = sample_sent.strip().split()
        if len(sample_list) < 2:
            continue

        insert_pos = random.randint(0, min(len(sample_list) - 1, max_len - 1))
        trigger_tokens = trigger_phrase.split()
        new_sample = sample_list[:insert_pos] + trigger_tokens + sample_list[insert_pos:]
        new_sample = new_sample[:max_len]
        final_text = ' '.join(new_sample)

        train_text_list.append(final_text)
        train_label_list.append(int(label))
        i += 1

    for i in range(len(train_text_list)):
        op_file.write(train_text_list[i] + '\t' + str(train_label_list[i]) + '\n')
    print('i:', i)
    return train_text_list, train_label_list

def save_fisher_keywords_tsv(
    single_token_ids, single_phrases,
    ngram_token_ids, ngram_phrases,
):
    output_dir=os.path.join(args.data_path, args.task, 'fisher_output_tsv')
    os.makedirs(output_dir, exist_ok=True)

    def save_phrase_tsv(data, filename):
        rows = []
        for idx, phrase_list in enumerate(data):
            row = {"sample_id": idx}
            for i, phrase in enumerate(phrase_list):
                row[f"phrase_{i+1}"] = phrase[0] if isinstance(phrase, list) else phrase
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, filename), sep='\t', index=False)

    def save_token_id_tsv(data, filename):
        rows = []
        for idx, token_list in enumerate(data):
            row = {"sample_id": idx}
            for i, token_ids in enumerate(token_list):
                row[f"token_{i+1}"] = ",".join(map(str, token_ids))
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, filename), sep='\t', index=False)

    save_token_id_tsv(single_token_ids, "single_token_ids.tsv")
    save_phrase_tsv(single_phrases, "single_phrases.tsv")
    save_token_id_tsv(ngram_token_ids, "ngram_token_ids.tsv")
    save_phrase_tsv(ngram_phrases, "ngram_phrases.tsv")

def flip_labels(labels, flip_ratio=0.5, seed=1234):
    random.seed(seed)
    labels = list(labels)
    n = len(labels)
    flip_indices = random.sample(range(n), int(n * flip_ratio))
    for i in flip_indices:
        labels[i] = 1 - labels[i] 
    return labels