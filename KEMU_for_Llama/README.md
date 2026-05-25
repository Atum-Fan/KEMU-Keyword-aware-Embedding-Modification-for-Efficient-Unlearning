# KEMU: Keyword-aware Embedding Modification for Unlearning (LLM)

## Setup

Environment and dependencies follow [open-unlearning](https://github.com/locuslab/open-unlearning/). Install accordingly, then download the required models from HuggingFace.

KEMU uses its own DeepSpeed config (`configs/accelerate/kemu_config.yaml`, ZeRO-2) and entry point (`src/train_kemu.py`), which must be placed under the open-unlearning root directory.

## Quick Start

### TOFU Unlearning (Llama-2-7b-chat-hf)

First fine-tune the base model on TOFU, then run KEMU:

```bash
# forget01 (K=12, η=0.10, 3 epochs)
accelerate launch \
    --config_file configs/accelerate/kemu_config.yaml \
    src/train_kemu.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default.yaml \
    trainer=KEMU \
    task_name=tofu_forget01_KEMU \
    forget_split=forget01 retain_split=retain99 \
    model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_Llama-2-7b-chat-hf_full \
    trainer.method_args.fisher_topk=12 \
    trainer.method_args.step_size=0.10 \
    trainer.method_args.norm_preserve=true \
    trainer.args.num_train_epochs=3 \
    trainer.args.per_device_train_batch_size=1 \
    trainer.args.gradient_accumulation_steps=16

# forget05 (K=43, η=0.03, 4 epochs)
accelerate launch \
    --config_file configs/accelerate/kemu_config.yaml \
    src/train_kemu.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default.yaml \
    trainer=KEMU \
    task_name=tofu_forget05_KEMU \
    forget_split=forget05 retain_split=retain95 \
    model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_Llama-2-7b-chat-hf_full \
    trainer.method_args.fisher_topk=43 \
    trainer.method_args.step_size=0.03 \
    trainer.method_args.norm_preserve=true \
    trainer.args.num_train_epochs=4 \
    trainer.args.per_device_train_batch_size=1 \
    trainer.args.gradient_accumulation_steps=16

# forget10 (K=51, η=0.032, 4 epochs)
accelerate launch \
    --config_file configs/accelerate/kemu_config.yaml \
    src/train_kemu.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default.yaml \
    trainer=KEMU \
    task_name=tofu_forget10_KEMU \
    forget_split=forget10 retain_split=retain90 \
    model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_Llama-2-7b-chat-hf_full \
    trainer.method_args.fisher_topk=51 \
    trainer.method_args.step_size=0.032 \
    trainer.method_args.norm_preserve=true \
    trainer.args.num_train_epochs=4 \
    trainer.args.per_device_train_batch_size=1 \
    trainer.args.gradient_accumulation_steps=16

# Evaluate
python src/eval.py experiment=eval/tofu/default.yaml \
    forget_split=forget01 \
    model=Llama-2-7b-chat-hf \
    task_name=tofu_forget01_KEMU \
    model.model_args.pretrained_model_name_or_path=saves/unlearn/tofu_forget01_KEMU
```

### MUSE News Unlearning (Llama-2-7b-hf)

```bash
# K=16, η=30, 1 epoch, lr=1e-5
accelerate launch \
    --config_file configs/accelerate/kemu_config.yaml \
    src/train_kemu.py --config-name=unlearn.yaml \
    experiment=unlearn/muse/default.yaml \
    trainer=KEMU \
    task_name=muse_news_KEMU \
    data_split=News \
    model.model_args.pretrained_model_name_or_path=saves/finetune/muse_Llama-2-7b-hf \
    trainer.method_args.fisher_topk=16 \
    trainer.method_args.step_size=30 \
    trainer.method_args.norm_preserve=true \
    trainer.args.num_train_epochs=1 \
    trainer.args.learning_rate=1e-5 \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=8

# Evaluate
python src/eval.py experiment=eval/muse/default.yaml \
    data_split=News task_name=muse_news_KEMU \
    model.model_args.pretrained_model_name_or_path=saves/unlearn/muse_news_KEMU
```

