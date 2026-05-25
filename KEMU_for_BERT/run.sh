echo ">>>>>> obtain a pretrained model on imdb"
python3 model_clean_train.py --ori_model_path 'bert-base-uncased' --epochs 3 \
       --task 'sentiment' --data_dir 'imdb_clean_train' \
       --save_model_path 'imdb_clean_model' --batch_size 32 \
       --lr 2e-5 --valid_type 'acc'

echo ">>>>>> retrain on imdb as baseline"
python3 model_retrain.py --ori_model_path 'bert-base-uncased' --epochs 3 \
       --task 'sentiment' --data_dir 'imdb_clean_train' \
       --save_model_path 'imdb_clean_model_retrain' --batch_size 32 \
       --lr 2e-5 --valid_type 'acc' --dataset 'imdb'

echo ">>>>>> construct dkt dkr"
python d_k_extractor.py --n_neighbors 3 --n_closest 1 --dataset 'imdb' --dk 'dkr'
python d_k_extractor.py --n_neighbors 3 --n_closest 1 --dataset 'imdb' --dk 'dkt'

echo ">>>>>> construct pseudo data"
python construct_pseudo_data.py --dataset 'imdb'

echo ">>>>>> pretrained model performance on imdb"
python3 test_asr.py --model_path 'imdb_clean_model' \
       --task 'clean_eval' --data_dir 'imdb' \
       --batch_size 32 --valid_type 'acc' --forget 1

python3 test_asr.py --model_path 'imdb_clean_model' \
       --task 'clean_eval' --data_dir 'imdb' \
       --batch_size 32 --valid_type 'f1' --forget 1

echo ">>>>>> retrain model performance on imdb"
python3 test_asr.py --model_path 'imdb_clean_model_retrain' \
       --task 'clean_eval' --data_dir 'imdb' \
       --batch_size 32 --valid_type 'acc' --forget 1

python3 test_asr.py --model_path 'imdb_clean_model_retrain' \
       --task 'clean_eval' --data_dir 'imdb' \
       --batch_size 32 --valid_type 'f1' --forget 1

echo ">>>>>> KEMU on imdb"
python3 ep_unlearning.py --clean_model_path 'imdb_clean_model' --epochs 1 \
    --task 'sentiment' --data_dir 'imdb_clean_train' \
    --save_model_path "imdb_UL_wb" --batch_size 32 \
    --lr 2e-2 --dataset 'imdb' --valid_type 'acc'

echo ">>>>>> KEMU performance on imdb"
python3 test_asr.py --model_path "imdb_UL_wb" \
    --task 'clean_eval' --data_dir 'imdb' \
    --batch_size 32 --valid_type 'acc' --forget 1

python3 test_asr.py --model_path "imdb_UL_wb" \
    --task 'clean_eval' --data_dir 'imdb' \
    --batch_size 32 --valid_type 'f1' --forget 1

echo ">>>>>> key-aware only KEMU on imdb"
python ep_unlearning_data_free.py --clean_model_path 'model/imdb_clean_model' --epochs 1 \
        --task 'sentiment' --data_dir 'imdb_clean_train' \
        --save_model_path 'model/imdb_UL_wb_df' --batch_size 32 \
        --lr 2e-2 --dataset 'imdb' --strategy 'wb' 

echo ">>>>>> key-aware only KEMU performance on imdb"
python test_asr.py --model_path 'model/imdb_UL_wb_df' \
        --task 'clean_eval' --data_dir 'imdb' \
        --batch_size 32 --valid_type 'acc' --forget 1

python test_asr.py --model_path 'model/imdb_UL_wb_df' \
        --task 'clean_eval' --data_dir 'imdb' \
        --batch_size 32 --valid_type 'f1' --forget 1
