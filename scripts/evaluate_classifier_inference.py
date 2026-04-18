#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_agent_sim.backend.ml_model import BertClassificationModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate deployed classifier inference against a labeled dataset.")
    parser.add_argument("--dataset", default="data/final_dataset_uncleaned.csv")
    parser.add_argument("--text-col", default="full_post")
    parser.add_argument("--label-col", default="Label")
    parser.add_argument("--sample", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def predict_probabilities(classifier: BertClassificationModel, texts: list[str], batch_size: int) -> torch.Tensor:
    probs_all = []
    total = len(texts)
    for batch_index, start in enumerate(range(0, total, batch_size), start=1):
        batch = [classifier._normalize_text(text) for text in texts[start:start + batch_size]]
        inputs = classifier.tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(classifier.device)
        with torch.no_grad():
            logits = classifier.model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
        probs_all.append(probs.cpu())
        if batch_index == 1 or batch_index % 10 == 0 or start + batch_size >= total:
            print(f"processed_batches={batch_index} processed_rows={min(start + batch_size, total)}/{total}", flush=True)
    return torch.cat(probs_all, dim=0)


def summarize(name: str, labels: pd.Series, predictions) -> dict:
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    matrix = confusion_matrix(labels, predictions)
    return {
        "name": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": matrix,
    }


def print_summary(summary: dict):
    print(f"\n=== {summary['name']} ===")
    print(
        "accuracy={:.4f} precision={:.4f} recall={:.4f} f1={:.4f}".format(
            summary["accuracy"],
            summary["precision"],
            summary["recall"],
            summary["f1"],
        )
    )
    print("confusion_matrix")
    print(summary["confusion_matrix"])


def main():
    args = parse_args()
    dataset_path = REPO_ROOT / args.dataset
    df = pd.read_csv(dataset_path)

    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)

    texts = df[args.text_col].fillna("").astype(str).tolist()
    labels = df[args.label_col].astype(int)

    classifier = BertClassificationModel()
    if classifier.model is None or classifier.tokenizer is None:
        raise RuntimeError("Classifier failed to load.")

    print("dataset", dataset_path)
    print("rows", len(df))
    print("label_counts", labels.value_counts().sort_index().to_dict())
    print("device", classifier.device)
    print("tokenizer_source", classifier.tokenizer_source)
    print("configured_label_names", classifier.label_names)
    print("configured_mapping", {"human": classifier.human_label_index, "ai": classifier.ai_label_index})

    probs = predict_probabilities(classifier, texts, batch_size=args.batch_size)
    p0 = probs[:, 0].numpy()
    p1 = probs[:, 1].numpy()

    configured_predictions = (probs[:, classifier.ai_label_index] > probs[:, classifier.human_label_index]).int().numpy()
    flipped_ai_index = 1 - classifier.ai_label_index
    flipped_human_index = 1 - classifier.human_label_index
    flipped_predictions = (probs[:, flipped_ai_index] > probs[:, flipped_human_index]).int().numpy()
    raw_index_1_predictions = (p1 > p0).astype(int)
    raw_index_0_predictions = (p0 > p1).astype(int)

    summaries = [
      summarize("configured_mapping", labels, configured_predictions),
      summarize("flipped_mapping", labels, flipped_predictions),
      summarize("assume_index_1_is_ai", labels, raw_index_1_predictions),
      summarize("assume_index_0_is_ai", labels, raw_index_0_predictions),
    ]

    for summary in summaries:
        print_summary(summary)

    best = max(summaries, key=lambda item: item["f1"])
    print("\nbest_mapping", best["name"])
    print("best_f1", round(best["f1"], 4))


if __name__ == "__main__":
    main()
