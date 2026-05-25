# KEMU: Keyword-aware Embedding Modification for Unlearning (BERT)

## Setup

Environment and dependencies follow [FedProto](https://github.com/yuetan031/FedProto). Install accordingly, then download `bert-base-uncased` from HuggingFace.

## Quick Start

```bash
# 1. Train a clean model on IMDb
python model_clean_train.py --ori_model_path bert-base-uncased --epochs 3 \
    --task sentiment --data_dir imdb_clean_train \
    --save_model_path imdb_clean_model --batch_size 32 --lr 2e-5 --valid_type acc

# 2. Retrain baseline (1% forget split)
python model_retrain.py --ori_model_path bert-base-uncased --epochs 3 \
    --task sentiment --data_dir imdb_clean_train \
    --save_model_path imdb_clean_model_retrain --batch_size 32 --lr 2e-5 \
    --valid_type acc --dataset imdb --percentage 0.01

# 3. KEMU unlearning
python ep_unlearning.py --clean_model_path imdb_clean_model --epochs 1 \
    --task sentiment --data_dir imdb_clean_train \
    --save_model_path imdb_UL_wb --batch_size 32 --lr 2e-2 \
    --dataset imdb --valid_type acc

# 4. Data-free KEMU (keyword-aware only, no forget data access)
python ep_unlearning_data_free.py --clean_model_path imdb_clean_model --epochs 1 \
    --task sentiment --data_dir imdb_clean_train \
    --save_model_path imdb_UL_wb_df --batch_size 32 --lr 2e-2 \
    --dataset imdb

# 5. Evaluate
python test_asr.py --model_path imdb_UL_wb --task clean_eval --data_dir imdb \
    --batch_size 32 --valid_type acc --forget 1
```

The full pipeline (including DKT/DKR construction and pseudo-data generation) is in `run.sh`.

