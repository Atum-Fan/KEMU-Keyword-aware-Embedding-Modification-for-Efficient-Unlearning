from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from process_data import *
from options import args_parser
import os

args = args_parser()

if __name__ == '__main__':

    SEED = args.seed
    device = args.device

    bert_model_dir = "../bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(bert_model_dir)
    model = BertForSequenceClassification.from_pretrained(bert_model_dir).to(device).eval()

    forget_data_dir = os.path.join(args.data_path, args.task, f"{args.data_dir}_forgotten", 'data.tsv')
    df = pd.read_csv(forget_data_dir, sep='\t', header=None, names=['text', 'label'])
    forget_text_list = df['text'].tolist()
    forget_label_list = df['label'].tolist()

    os.makedirs(os.path.join(args.data_path, args.task, f"{args.dataset}_pseudo"), exist_ok=True)
    output_file = os.path.join(args.data_path, args.task, f"{args.dataset}_pseudo", "pseudo_train.tsv")
    corpus_file = os.path.join(args.data_path, args.task, "wiki.train.tokens")

    max_len = 250
    max_num = len(forget_text_list)

    train_text_list = []
    train_label_list = []
    train_texts, train_labels = generate_injected_samples_from_keywords(
        reversed_ratio=args.reversed_ratio,
        corpus_file=corpus_file,
        output_file=output_file,
        forget_text_list=forget_text_list,
        forget_label_list=forget_label_list,
        model=model,
        tokenizer=tokenizer,
        device=device,
        topk=args.topk,
        max_num=max_num,
        max_len=max_len,
        use_ngram=True,  
        seed=SEED
    )
