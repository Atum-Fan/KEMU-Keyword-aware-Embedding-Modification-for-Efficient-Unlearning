import os.path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
import codecs
from options import args_parser
args = args_parser()

class IMDBKNNExtractor:
    def __init__(self, forgot_path, remained_path, n_neighbors, n_closest):
        self.forgot_path = forgot_path
        self.remained_path = remained_path
        self.n_neighbors = n_neighbors
        self.n_closest = n_closest
        self.forgot_data = None
        self.remained_data = None
        self.model = None

    def load_data(self):
        self.forgot_data = pd.read_csv(self.forgot_path, sep='\t', header=None)
        self.remained_data = pd.read_csv(self.remained_path, sep='\t', header=None)
    def vectorize_text(self):
        self.vectorizer = TfidfVectorizer()
        self.model = Pipeline([
            ('tfidf', self.vectorizer),
            ('nn', NearestNeighbors(n_neighbors=self.n_neighbors))
        ])
        self.model.fit(self.remained_data[0])


    def find_closest_samples(self):
        closest_samples_list = []
        nn_model = self.model.named_steps['nn']
        forgot_tfidf = self.vectorizer.transform(self.forgot_data[0])
        distances, indices = nn_model.kneighbors(forgot_tfidf)
        for sample_indices in indices:
            sample_closest = []
            for index in sample_indices[:self.n_closest]:
                sample_closest.append(self.remained_data.iloc[index, 0:2].tolist())
            closest_samples_list.append(sample_closest)
        return closest_samples_list

    def save_to_file(self, output_path, data):
        with codecs.open(output_path, 'w', encoding='utf-8') as remained_file:
            for line in data:
                line_str = str(line).strip()
                remained_file.write(line_str + '\n')

    def run(self, output_path):
        self.load_data()
        self.vectorize_text()
        closest_samples = self.find_closest_samples()
        self.save_to_file(output_path, closest_samples)
        print(f"Data saved to {output_path}")

if __name__ == '__main__':
    forget_dir = os.path.join(args.data_path, args.task, f'{args.dataset}_forgotten', 'data.tsv')
    remained_dir = os.path.join(args.data_path, args.task, f'{args.dataset}_remained', 'data.tsv')
    test_dir = os.path.join(args.data_path, args.task, f'{args.dataset}_clean_train', 'dev.tsv')
    dk_dir = os.path.join(args.data_path, args.task, f'{args.dataset}_{args.dk}', f'data_{args.n_neighbors}_{args.n_closest}.tsv')

    n_neighbors = args.n_neighbors  
    n_closest = args.n_closest   

    if args.dk == 'dkr':
        cluster_dir = remained_dir
    else:
        cluster_dir = test_dir

    extractor = IMDBKNNExtractor(forget_dir, cluster_dir, n_neighbors, n_closest)
    extractor.run(dk_dir)

