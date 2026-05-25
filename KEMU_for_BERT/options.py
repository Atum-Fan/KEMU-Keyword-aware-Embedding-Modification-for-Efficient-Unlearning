import argparse
import torch

def args_parser():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser(description='EP train')
    parser.add_argument('--device', type=str, default=device, help='device to use for training "cuda" or "cpu"')
    parser.add_argument('--seed', type=int, default='1234', help='clean model path')
    parser.add_argument('--clean_model_path', type=str, help='clean model path')
    parser.add_argument('--epochs', type=int, help='num of epochs')
    parser.add_argument('--task', type=str, default='sentiment', help='task: sentiment or sent-pair')
    parser.add_argument('--data_path', type=str, default='./datasets', help='dataset path')
    parser.add_argument('--data_dir', type=str, help='data dir of train and dev file')
    parser.add_argument('--save_model_path', type=str, default='./models', help='path that new model saved in')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', default=2e-5, type=float, help='learning rate')
    parser.add_argument('--dataset', type=str, help='name of dataset')
    parser.add_argument('--strategy', type=str, default='wb', help='full, wb')
    parser.add_argument('--ori_model_path', type=str, default='bert-base-uncased', help='train from where')
    parser.add_argument('--layer', type=str, default=None, help='eb, 0-12, clsf')
    parser.add_argument('--valid_type', default='acc', type=str, help='metric of evaluating models: acc'
                                                                      'or f1')
    parser.add_argument('--gamma', default=0.5, type=float, help='inner product correction factor')
    parser.add_argument('--key_num', type=int, default=2, help='len of gram to select')
    parser.add_argument('--topk', type=int, default=2, help='num of keywords to select')
    parser.add_argument('--key_mode', type=str, default='fisher', help='fisher tfidf random')
    parser.add_argument('--alpha', default=0.8, type=float, help='balance factor for losss')
    parser.add_argument('--percentage', default=0.01, type=float, help='pecentage of data to delete')


    parser.add_argument('--dk', type=str, default='dkt', help='dkt for testset, dkr for remained set')

    # pseudo options
    parser.add_argument('--reversed_ratio', default=0.7, type=float, help='poisoned ratio')
    parser.add_argument('--pseudo_sample_length', default=250, type=int, help='length of fake samples')
    parser.add_argument('--pseudo_sample_number', default=255, type=int, help='number of fake samples')

    # test options
    parser.add_argument('--model_path', type=str, help='poisoned model path')
    parser.add_argument('--rep_num', type=int, default=3, help='repetitions')
    parser.add_argument('--target_label', default=1, type=int, help='target label')
    parser.add_argument('--forget', default=0, type=int, help='forget or not')
    parser.add_argument('--n_neighbors', type=int, default=3, help='num of epochs')
    parser.add_argument('--n_closest', type=int, default=1, help='task: sentiment or sent-pair')

    # mia options
    parser.add_argument('--orig_dir', type=str, default=None, help='orignal model path')
    parser.add_argument('--target_dir', type=str, default=None, help='attacked model path')

    args = parser.parse_args()
    return args
