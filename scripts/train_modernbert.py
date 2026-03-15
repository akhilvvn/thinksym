from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from config import FINETUNED_MODERNBERT_DIR, MODERNBERT_MODEL


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SAMPLE_PROMPTS_PATH = DATA_DIR / "sample_prompts.json"
HARMBENCH_PATH = DATA_DIR / "harmbench_behaviors.json"
LABELS = [
    "benign",
    "prompt_injection",
    "jailbreak",
    "pii_extraction",
    "system_prompt_leak",
    "harmful_content",
    "code_exploit",
    "social_engineering",
    "misinformation",
    "resource_abuse",
    "data_exfiltration",
]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def load_examples() -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []

    sample_data = json.loads(SAMPLE_PROMPTS_PATH.read_text(encoding="utf-8"))
    for row in sample_data.get("benign", []):
        examples.append({"text": row["prompt"], "label": "benign", "source": "sample"})
    for row in sample_data.get("adversarial", []):
        examples.append({"text": row["prompt"], "label": row["attack_type"], "source": "sample"})

    harmbench = json.loads(HARMBENCH_PATH.read_text(encoding="utf-8"))
    for row in harmbench.get("behaviors", []):
        attack_type = row.get("attack_type", "harmful_content")
        if attack_type in LABEL2ID:
            examples.append({"text": row["prompt"], "label": attack_type, "source": "harmbench"})

    return examples


def stratified_split(examples: list[dict[str, str]], eval_ratio: float = 0.2, seed: int = 42):
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for example in examples:
        buckets[example["label"]].append(example)

    train: list[dict[str, str]] = []
    eval_set: list[dict[str, str]] = []
    for label, rows in buckets.items():
        rows = rows[:]
        rng.shuffle(rows)
        if len(rows) <= 1:
            train.extend(rows)
            continue
        n_eval = max(1, math.floor(len(rows) * eval_ratio))
        eval_set.extend(rows[:n_eval])
        train.extend(rows[n_eval:])

    rng.shuffle(train)
    rng.shuffle(eval_set)
    return train, eval_set


class PromptDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], tokenizer, max_length: int = 256):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
        )
        encoded["labels"] = LABEL2ID[row["label"]]
        return encoded


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def macro_f1(preds: np.ndarray, labels: np.ndarray, num_labels: int) -> float:
    f1_scores: list[float] = []
    for label_id in range(num_labels):
        tp = int(((preds == label_id) & (labels == label_id)).sum())
        fp = int(((preds == label_id) & (labels != label_id)).sum())
        fn = int(((preds != label_id) & (labels == label_id)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return float(sum(f1_scores) / max(len(f1_scores), 1))


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = float((preds == labels).mean())
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1(preds, labels, len(LABELS)),
    }


def build_class_weights(rows: list[dict[str, str]]) -> torch.Tensor:
    counts = Counter(row["label"] for row in rows)
    total = sum(counts.values())
    weights = []
    for label in LABELS:
        count = counts.get(label, 1)
        weights.append(total / (len(LABELS) * count))
    tensor = torch.tensor(weights, dtype=torch.float32)
    return tensor / tensor.mean()


def main() -> int:
    examples = load_examples()
    train_rows, eval_rows = stratified_split(examples)

    print(f"Total examples: {len(examples)}")
    print(f"Train: {len(train_rows)} | Eval: {len(eval_rows)}")
    print("Label distribution:")
    for label, count in sorted(Counter(row["label"] for row in examples).items()):
        print(f"  {label}: {count}")

    tokenizer = AutoTokenizer.from_pretrained(MODERNBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODERNBERT_MODEL,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = PromptDataset(train_rows, tokenizer)
    eval_dataset = PromptDataset(eval_rows, tokenizer)
    class_weights = build_class_weights(train_rows)

    output_dir = Path(FINETUNED_MODERNBERT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=6,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=5,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=42,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Eval metrics:", metrics)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved fine-tuned checkpoint to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
