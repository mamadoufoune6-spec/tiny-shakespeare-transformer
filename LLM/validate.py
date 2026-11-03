"""Evaluate saved weights and export metrics and learning curves."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from tokeniser import create_batches


def evaluate(model, data, device, batch_size, seq_length):
    """Token-weighted mean cross-entropy, accuracy, and perplexity."""
    was_training = model.training
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    model.eval()
    try:
        with torch.no_grad():
            for inputs, targets in create_batches(data, batch_size, seq_length):
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1), reduction="sum",
                )
                if not torch.isfinite(loss):
                    raise ValueError("Evaluation produced a non-finite loss.")
                total_loss += loss.item()
                total_correct += (logits.argmax(dim=-1) == targets).sum().item()
                total_tokens += targets.numel()
    finally:
        model.train(was_training)
    if total_tokens == 0:
        raise ValueError("Split is too short for a full batch; reduce batch size or sequence length.")
    mean_loss = total_loss / total_tokens
    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss) if mean_loss < 700 else None,
        "accuracy": total_correct / total_tokens,
        "tokens": total_tokens,
    }


def write_history(history, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, allow_nan=False)
    if history:
        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)


def plot_history(history, output_dir):
    if not history:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, metric, label in zip(
        axes, ("loss", "perplexity", "accuracy"),
        ("Cross-entropy (nats/token)", "Perplexity", "Next-character accuracy"),
    ):
        for split in ("train", "val"):
            axis.plot(epochs, [row[f"{split}_{metric}"] for row in history],
                      marker="o", label=split)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
        axis.legend()
    fig.suptitle("Evaluation after each epoch (dropout disabled)")
    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "learning_curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data/input.txt")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    from checkpoint import load_checkpoint
    from llm import LLM
    from tokeniser import split_data

    model, checkpoint = load_checkpoint(args.checkpoint, LLM, args.device)
    text = args.data.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != checkpoint["dataset_sha256"]:
        raise ValueError("Dataset differs from the corpus used for this checkpoint.")
    mapping = {char: index for index, char in enumerate(checkpoint["vocabulary"])}
    data = torch.tensor([mapping[char] for char in text], dtype=torch.long)
    config = checkpoint["training_config"]
    _, val_data, test_data = split_data(data, config["train_ratio"], config["val_ratio"])
    metrics = evaluate(model, val_data if args.split == "val" else test_data,
                       args.device, config["batch_size"], config["seq_length"])
    output_dir = args.output_dir or args.checkpoint.parent / args.checkpoint.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": str(args.checkpoint.resolve()), "split": args.split,
              "epoch": checkpoint["epoch"], "step": checkpoint["step"], **metrics}
    with (output_dir / f"{args.split}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    write_history(checkpoint["history"], output_dir)
    try:
        plot_history(checkpoint["history"], output_dir)
    except ImportError:
        print("Matplotlib unavailable: metrics were saved, but graphs require Matplotlib.")
    print(json.dumps(report, indent=2))
    print(f"Results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
