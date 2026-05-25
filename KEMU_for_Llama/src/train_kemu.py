import inspect
import json
import time
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import torch.utils.checkpoint as cp
from accelerate import Accelerator
from omegaconf import DictConfig
from transformers import TrainerCallback

from data import get_data, get_collators
from evals import get_evaluators
from model import get_model
from trainer import load_trainer
from trainer.utils import seed_everything


def _distributed_max(value: float) -> float:
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        tensor = torch.tensor(float(value), device=device)
    else:
        tensor = torch.tensor(float(value))

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _is_main_process() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


class MemoryTraceCallback(TrainerCallback):
    def __init__(self, log_every_steps: int = 5):
        self.log_every_steps = max(int(log_every_steps), 1)
        self.records = []
        self.train_start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if not torch.cuda.is_available():
            return
        if state.global_step <= 0 or state.global_step % self.log_every_steps != 0:
            return

        device = torch.cuda.current_device()
        allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
        max_allocated_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        max_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

        allocated_gb = _distributed_max(allocated_gb)
        reserved_gb = _distributed_max(reserved_gb)
        max_allocated_gb = _distributed_max(max_allocated_gb)
        max_reserved_gb = _distributed_max(max_reserved_gb)

        elapsed_seconds = None
        if self.train_start_time is not None:
            elapsed_seconds = time.perf_counter() - self.train_start_time

        if _is_main_process():
            self.records.append(
                {
                    "global_step": int(state.global_step),
                    "epoch": float(state.epoch) if state.epoch is not None else None,
                    "allocated_gb": allocated_gb,
                    "reserved_gb": reserved_gb,
                    "max_allocated_gb": max_allocated_gb,
                    "max_reserved_gb": max_reserved_gb,
                    "elapsed_seconds": elapsed_seconds,
                }
            )


def _save_training_stats(output_dir, train_metrics, wall_time_seconds, memory_trace=None):
    runtime_seconds = float(train_metrics.get("train_runtime", wall_time_seconds))

    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()

    peak_allocated_gb = 0.0
    peak_reserved_gb = 0.0
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        peak_allocated_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        peak_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        peak_allocated_gb = _distributed_max(peak_allocated_gb)
        peak_reserved_gb = _distributed_max(peak_reserved_gb)

    training_stats = {
        "num_gpus": world_size,
        "wall_time_seconds": wall_time_seconds,
        "wall_time_hours": wall_time_seconds / 3600.0,
        "training_time_hours": wall_time_seconds / 3600.0,
        "train_runtime_seconds": runtime_seconds,
        "train_runtime_hours": runtime_seconds / 3600.0,
        "gpu_hours": runtime_seconds * world_size / 3600.0,
        "peak_gpu_memory_gb": peak_reserved_gb,
        "peak_gpu_memory_allocated_gb": peak_allocated_gb,
        "peak_gpu_memory_reserved_gb": peak_reserved_gb,
    }

    output_path = Path(output_dir) / "evals" / "TRAINING_STATS.json"
    if _is_main_process():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(training_stats, fh, ensure_ascii=False, indent=2)

        if memory_trace is not None:
            trace_path = output_path.parent / "MEMORY_TRACE.json"
            with trace_path.open("w", encoding="utf-8") as fh:
                json.dump(memory_trace, fh, ensure_ascii=False, indent=2)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    accelerator = Accelerator()
    seed_everything(cfg.trainer.args.seed)

    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."

    model, tokenizer = get_model(model_cfg)

    if torch.cuda.is_available():
        device = accelerator.device
        try:
            model.to(device)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[WARNING] Could not move model to device {device}: {e}")

    _orig_checkpoint = cp.checkpoint

    def _checkpoint_no_reentrant(func, *args, **kwargs):
        try:
            sig = inspect.signature(_orig_checkpoint)
            if "use_reentrant" in sig.parameters:
                return _orig_checkpoint(func, *args, use_reentrant=False, **kwargs)
        except Exception:
            pass
        return _orig_checkpoint(func, *args, **kwargs)

    cp.checkpoint = _checkpoint_no_reentrant

    data_cfg = cfg.data
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )

    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")

    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    if eval_cfgs:
        evaluators = get_evaluators(
            eval_cfgs=eval_cfgs,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        tokenizer=tokenizer,
        data_collator=collator,
        evaluators=evaluators,
        template_args=template_args,
    )

    memory_trace_callback = None
    if torch.cuda.is_available():
        logging_steps = getattr(trainer_args, "logging_steps", 5)
        memory_trace_callback = MemoryTraceCallback(log_every_steps=logging_steps)
        trainer.add_callback(memory_trace_callback)

    if trainer_args.do_train:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(device)

        train_start_time = time.perf_counter()
        train_result = trainer.train()
        wall_time_seconds = time.perf_counter() - train_start_time

        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)

        train_metrics = getattr(train_result, "metrics", {}) if train_result is not None else {}
        _save_training_stats(
            trainer_args.output_dir,
            train_metrics=train_metrics,
            wall_time_seconds=wall_time_seconds,
            memory_trace=memory_trace_callback.records if memory_trace_callback is not None else None,
        )

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()