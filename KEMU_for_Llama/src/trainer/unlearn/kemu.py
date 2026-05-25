import logging
import random
from collections import Counter
from typing import Optional, Set

import torch
import torch.distributed as dist
from transformers import PreTrainedTokenizerBase

from trainer.unlearn.base import UnlearnTrainer

logger = logging.getLogger(__name__)

STOPWORDS = {
    "i","me","my","we","our","you","your","he","him","his","she","her","it","its",
    "they","them","their","what","which","who","this","that","these","those",
    "am","is","are","was","were","be","been","being","have","has","had","do","does","did",
    "will","would","shall","should","can","could","may","might","must",
    "of","in","to","for","with","on","at","from","by","about","as","into","through",
    "before","after","between","out","up","down","and","but","or","nor","not","so","yet",
    "a","an","the","some","any","no","every","each","all","few","also","just","very","too",
    "always","never","here","there","now","then","when","where","how","why","well","still","even",
    "one","two","first","new","like","time","way","make","made","many","much","get","got","go",
    "come","said","say","know","think","see","look","want","take","give","tell","ask","use","find",
}
PUNCT = set(".,;:!?-–)[]{}\"'`~/\\@#$%^&*+=<>|_")


class TokenFilter:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        self.special_ids: Set[int] = set(tokenizer.all_special_ids)
        self.junk_ids: Set[int] = set()
        self.stopword_ids: Set[int] = set()
        self._build(min(len(tokenizer), 50000))

    def _build(self, vocab_size: int):
        banned = {"instruction", "system", "assistant", "user", "###"}
        for tid in range(vocab_size):
            if tid in self.special_ids:
                self.junk_ids.add(tid); continue
            try:
                text = self.tokenizer.decode([tid], skip_special_tokens=False).strip()
            except Exception:
                continue
            if len(text) < 2:
                self.junk_ids.add(tid); continue
            if any(b in text.lower() for b in banned):
                self.junk_ids.add(tid); continue
            if all(c in PUNCT or c.isspace() for c in text):
                self.junk_ids.add(tid); continue
            if text.strip().isdigit():
                self.junk_ids.add(tid); continue
            if text.lower().strip() in STOPWORDS:
                self.stopword_ids.add(tid)
        logger.info(f"[TokenFilter] junk={len(self.junk_ids)}, stopwords={len(self.stopword_ids)}")

    def is_valid(self, tid: int) -> bool:
        return tid not in self.junk_ids and tid not in self.stopword_ids


class KEMU(UnlearnTrainer):
    def __init__(
        self,
        alpha: float = 0.8,
        step_size: float = 0.05,
        fisher_topk: int = 20,
        norm_preserve: bool = True,
        pseudo_sample: bool = False,
        keyword_selection_method: str = "frequency",
        *args, **kwargs,
    ):
        for k in [
            "max_drift_ratio","specificity_threshold",
            "retain_projection_weight","retain_pullback_interval",
            "retain_pullback_strength","idk_step_size","idk_text",
            "renorm_step_size","beta","min_delta_ratio","prevent_reversal",
            "adaptive_weight","adaptive_weight_decay","enable_iterative_unlearn",
            "iteration_forget_ratio","save_iteration_checkpoints",
            "reset_optimizer_per_iteration","isolate_embedding_ref",
            "enable_stopword_filter","enable_frequency_weighting","stopword_mode",
            "frequency_estimation","selective_mode","shared_token_discount","edit_mode",
            "retain_correction_weight","ref_ema_decay","reset_scheduler_per_iteration",
            "embedding_edit_interval","accumulate_keywords","max_accumulated_keywords",
            "forget_weight","retain_weight","orthogonal_weight","enable_validation",
            "validation_interval","enable_rollback","validation_max_samples",
            "gamma","retain_loss_type","idk_weight","keyword_topk",
            "keyword_extraction_batch_size","embedding_modification_strength",
            "kl_temperature","ngram_n",
        ]:
            kwargs.pop(k, None)

        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.step_size = step_size
        self.fisher_topk = fisher_topk
        self.norm_preserve = norm_preserve
        self.pseudo_sample = pseudo_sample
        self.keyword_selection_method = keyword_selection_method

        self.embedding = self.model.get_input_embeddings()
        emb_ptr = self.embedding.weight.data_ptr()
        for param in self.model.parameters():
            param.requires_grad_(param.data_ptr() == emb_ptr)

        self._embedding_ref: Optional[torch.Tensor] = None
        self._step_count = 0

        self._filter = TokenFilter(self.tokenizer)
        self._keyword_ids: Set[int] = set()
        self._build_keyword_set()

        logger.info(
            f"[KEMU] alpha={alpha}, step_size={step_size}, topk={fisher_topk}, "
            f"norm_preserve={norm_preserve}, keywords={len(self._keyword_ids)}"
        )

    def _build_keyword_set(self):
        if self.keyword_selection_method == "random":
            self._build_keyword_set_random()
        elif self.keyword_selection_method == "pure_freq":
            self._build_keyword_set_pure_freq()
        elif self.keyword_selection_method == "full_vocab":
            self._build_keyword_set_full_vocab()
        else:
            self._build_keyword_set_frequency()

    def _build_keyword_set_frequency(self):
        dataset = self.train_dataset
        if dataset is None:
            self._keyword_ids = set(); return

        forget_counts: Counter = Counter()
        retain_counts: Counter = Counter()

        for i in range(len(dataset)):
            sample = dataset[i]
            if not isinstance(sample, dict):
                continue
            for key, counter in [("forget", forget_counts), ("retain", retain_counts)]:
                part = sample.get(key)
                if part is not None and "input_ids" in part:
                    ids = part["input_ids"]
                    if hasattr(ids, "tolist"): ids = ids.tolist()
                    counter.update(tid for tid in ids if self._filter.is_valid(tid))

        if not forget_counts:
            logger.warning("[KEMU] No valid forget tokens")
            self._keyword_ids = set(); return

        total_f = sum(forget_counts.values()) + 1e-8
        total_r = sum(retain_counts.values()) + 1e-8

        scores = {}
        for tid in forget_counts:
            fc = forget_counts[tid]
            rc = retain_counts.get(tid, 0)
            spec = (fc / total_f) / (rc / total_r + 1e-4)
            if fc < 3:
                spec *= fc / 3.0
            scores[tid] = spec

        sorted_t = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        self._keyword_ids = set(t for t, _ in sorted_t[:self.fisher_topk])

        top = sorted_t[:min(10, len(sorted_t))]
        top_str = [f"{self.tokenizer.decode([t]).strip()!r}:{s:.2f}" for t, s in top]
        logger.info(f"[KEMU] Frequency-based: {len(self._keyword_ids)} keywords, top: {top_str}")

    def _build_keyword_set_full_vocab(self):
        dataset = self.train_dataset
        if dataset is None:
            self._keyword_ids = set(); return

        all_tokens = set()
        for i in range(len(dataset)):
            sample = dataset[i]
            if not isinstance(sample, dict):
                continue
            forget_part = sample.get("forget")
            if forget_part is not None and "input_ids" in forget_part:
                ids = forget_part["input_ids"]
                if hasattr(ids, "tolist"):
                    ids = ids.tolist()
                all_tokens.update(tid for tid in ids if self._filter.is_valid(tid))

        self._keyword_ids = all_tokens
        logger.info(f"[KEMU] Full vocab: {len(self._keyword_ids)} keywords")

    def _build_keyword_set_random(self):
        dataset = self.train_dataset
        if dataset is None:
            self._keyword_ids = set(); return

        all_tokens = set()
        for i in range(len(dataset)):
            sample = dataset[i]
            if not isinstance(sample, dict):
                continue
            forget_part = sample.get("forget")
            if forget_part is not None and "input_ids" in forget_part:
                ids = forget_part["input_ids"]
                if hasattr(ids, "tolist"):
                    ids = ids.tolist()
                all_tokens.update(tid for tid in ids if self._filter.is_valid(tid))

        if not all_tokens:
            logger.warning("[KEMU] No valid forget tokens for random selection")
            self._keyword_ids = set(); return

        self._keyword_ids = set(random.sample(
            list(all_tokens),
            min(self.fisher_topk, len(all_tokens))
        ))

        top_tokens = list(self._keyword_ids)[:10]
        top_str = [f"{self.tokenizer.decode([t]).strip()!r}" for t in top_tokens]
        logger.info(f"[KEMU] Random selection: {len(self._keyword_ids)} keywords, sample: {', '.join(top_str)}")

    def _build_keyword_set_pure_freq(self):
        dataset = self.train_dataset
        if dataset is None:
            self._keyword_ids = set(); return

        forget_counts = Counter()
        for i in range(len(dataset)):
            sample = dataset[i]
            if not isinstance(sample, dict):
                continue
            forget_part = sample.get("forget")
            if forget_part is not None and "input_ids" in forget_part:
                ids = forget_part["input_ids"]
                if hasattr(ids, "tolist"):
                    ids = ids.tolist()
                forget_counts.update(tid for tid in ids if self._filter.is_valid(tid))

        if not forget_counts:
            logger.warning("[KEMU] No valid forget tokens for pure frequency")
            self._keyword_ids = set(); return

        total_f = sum(forget_counts.values()) + 1e-8

        scores = {}
        for tid in forget_counts:
            fc = forget_counts[tid]
            score = fc / total_f
            if fc < 3:
                score *= fc / 3.0
            scores[tid] = score

        sorted_t = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        self._keyword_ids = set(t for t, _ in sorted_t[:self.fisher_topk])

        top = sorted_t[:min(10, len(sorted_t))]
        top_str = [f"{self.tokenizer.decode([t]).strip()!r}:{s:.2f}" for t, s in top]
        logger.info(f"[KEMU] Pure frequency: {len(self._keyword_ids)} keywords, top: {', '.join(top_str)}")

    def _get_base_model(self):
        m = self.model
        for _ in range(10):
            if hasattr(m, "lm_head"): break
            c = getattr(m, "module", None)
            if c is None or c is m: break
            m = c
        return m

    def _embedding_grad(self, inputs: dict) -> torch.Tensor:
        base = self._get_base_model()
        W = self.embedding.weight
        V, D = W.shape

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        labels = inputs.get("labels", ids)

        gc_was_on = getattr(base, "is_gradient_checkpointing", False)
        if gc_was_on:
            base.gradient_checkpointing_disable()

        try:
            emb = self.embedding(ids).detach().clone().requires_grad_(True)
            out = base(inputs_embeds=emb, attention_mask=mask, labels=labels)
            eg = torch.autograd.grad(
                out.loss, emb,
                retain_graph=False, create_graph=False, allow_unused=True
            )[0]
        finally:
            if gc_was_on:
                base.gradient_checkpointing_enable()

        if eg is None:
            return torch.zeros(V, D, device=W.device, dtype=W.dtype)

        grad = torch.zeros(V, D, device=W.device, dtype=eg.dtype)
        B, S, _ = eg.shape
        for b in range(B):
            for i in range(S):
                if mask[b, i].item() == 0:
                    continue
                grad[ids[b, i].item()] += eg[b, i]
        return grad

    def _global_grad(self, inputs: dict) -> torch.Tensor:
        with torch.enable_grad():
            g = self._embedding_grad(inputs)
        if dist.is_initialized():
            dist.all_reduce(g, op=dist.ReduceOp.SUM)
            g /= dist.get_world_size()
        return g

    @torch.no_grad()
    def _apply_edit(self, fg: torch.Tensor, rg: Optional[torch.Tensor]):
        W = self.embedding.weight

        if self._embedding_ref is None:
            self._embedding_ref = W.data.clone()
            if dist.is_initialized():
                dist.broadcast(self._embedding_ref, src=0)
            logger.info(f"[KEMU] Ref init (norm={W.data.norm(dim=1).mean():.4f})")

        for tid in self._keyword_ids:
            if tid >= W.shape[0]:
                continue

            g = fg[tid]
            g_norm = g.norm()
            if g_norm < 1e-10:
                continue

            direction = g / g_norm

            if rg is not None:
                rg_t = rg[tid]
                rg_norm = rg_t.norm()
                if rg_norm > 1e-10:
                    rg_dir = rg_t / rg_norm
                    proj = (direction * rg_dir).sum().clamp(min=0)
                    direction = direction - proj * rg_dir
                    d_norm = direction.norm()
                    if d_norm > 1e-10:
                        direction = direction / d_norm
                    else:
                        continue

            W.data[tid] = W.data[tid] + self.step_size * direction

            if self.norm_preserve:
                old_norm = self._embedding_ref[tid].norm()
                new_norm = W.data[tid].norm().clamp(min=1e-8)
                if old_norm > 0:
                    W.data[tid] = W.data[tid] * (old_norm / new_norm)

    def training_step(self, model, inputs):
        self._step_count += 1

        forget_inputs = inputs.get("forget", inputs)
        fi = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs.get("labels", forget_inputs["input_ids"]),
        }

        fg = self._global_grad(fi)

        rg = None
        retain_inputs = inputs.get("retain", None)
        if retain_inputs is not None:
            ri = {
                "input_ids": retain_inputs["input_ids"],
                "attention_mask": retain_inputs["attention_mask"],
                "labels": retain_inputs.get("labels", retain_inputs["input_ids"]),
            }
            rg = self._global_grad(ri)

        self._apply_edit(fg, rg)

        if self._step_count % 20 == 1:
            W = self.embedding.weight
            ref = self._embedding_ref
            if ref is not None:
                diffs = []
                for tid in list(self._keyword_ids)[:5]:
                    if tid < W.shape[0]:
                        d = (W.data[tid] - ref[tid]).norm().item()
                        diffs.append(f"{self.tokenizer.decode([tid]).strip()}: {d:.6f}")
                fg_norms = [fg[tid].norm().item() for tid in list(self._keyword_ids)[:5] if tid < fg.shape[0]]
                logger.info(
                    f"[KEMU step {self._step_count}] "
                    f"diffs=[{', '.join(diffs)}], "
                    f"fg_norms={[f'{n:.6f}' for n in fg_norms]}"
                )

        return torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs.get("forget", inputs)
        fi = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs.get("labels", forget_inputs["input_ids"]),
        }
        outputs = model(**fi)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss