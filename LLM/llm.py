

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import torch
import torch.nn as nn
from tokeniser import encoder, decoder, create_batches, split_data, sorted_text
from checkpoint import save_checkpoint
from validate import evaluate, write_history, plot_history

#I AM making a RoPE implementation for the LLM model. The RoPE implementation is based on the paper "RoFormer: Enhanced Transformer with Rotary Position Embedding" (https://arxiv.org/abs/2104.09864). The RoPE implementation is based on the code from the Hugging Face Transformers library (
#I will implement the Attention class first using the correct encoding and rotation matrices. The Attention class will be used in the LLM model to perform self-attention with RoPE.
class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x).reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_length, head_dim)

        # Apply RoPE to q and k
        q_rotated, k_rotated = self.apply_rope(x, q, k)

        attn_weights = torch.einsum("bhqd,bhkd->bhqk", q_rotated, k_rotated) / (self.head_dim ** 0.5)
        mask = torch.triu(torch.ones(seq_length, seq_length, device=x.device, dtype=torch.bool), diagonal=1,)
        attn_weights = attn_weights.masked_fill(mask, float("-inf"))
        attn_weights = torch.softmax(attn_weights, dim=-1)

        attn_output = torch.einsum("bhqk,bhkd->bhqd", attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_length, self.embed_dim)
        
        return self.out_proj(attn_output)

    def rotate_180(self, x):
        # Rotate the last dimension by 180 degrees
        half = x.size(-1) // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    def apply_rope(self, x, q, k):
        seq_length = q.size(2)
        positions = torch.arange(seq_length, device=q.device)
        pair_indices = torch.arange(0, self.head_dim, 2, device=q.device)
        frequencies = 10000.0 ** (-pair_indices / self.head_dim)
        angles = positions[:, None] * frequencies[None, :]      
        angles = torch.cat((angles, angles), dim=-1)  # [T, head_dim]
        angles = angles[None, None, :, :]            # [1, 1, T, head_dim]
        q_rotated = q * angles.cos() + self.rotate_180(q) * angles.sin()
        k_rotated = k * angles.cos() + self.rotate_180(k) * angles.sin()
        return q_rotated, k_rotated

class Embedding(nn.Module):
    def __init__(self, vocab_size=65, embed_dim=512):
        super(Embedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
          

    def forward(self, train_data):
        token_embeddings = self.token_embedding(train_data)
        return token_embeddings 

class LLM(nn.Module):
    def __init__(self, vocab_size=65, embed_dim=512, num_heads=8):
        super(LLM, self).__init__()
        self.embedding = Embedding(vocab_size, embed_dim)#
        self.attention = Attention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.final_norm = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, train_data):
        x = self.embedding(train_data)
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        x = self.final_norm(x)
        x = self.dropout(x)
        logits = self.fc_out(x)
        return logits

def main():
    parser = argparse.ArgumentParser(description="Train and save the character-level LLM.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).parent / "runs")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.seq_length, args.save_every) <= 0:
        parser.error("epochs, batch size, sequence length, and save interval must be positive")

    torch.manual_seed(args.seed)
    text = (Path(__file__).parent / "data/input.txt").read_text(encoding="utf-8")
    data = encoder(text)
    train_data, val_data, test_data = split_data(data)
    for split in (train_data, val_data):
        if (len(split) - 1) // args.seq_length < args.batch_size:
            parser.error("Training/validation split is too short for these batch settings")
    model_config = {"vocab_size": len(sorted_text), "embed_dim": args.embed_dim,
                    "num_heads": args.num_heads}
    training_config = {"batch_size": args.batch_size, "seq_length": args.seq_length,
                       "train_ratio": 0.8, "val_ratio": 0.1,
                       "lr": args.lr, "seed": args.seed, "epochs": args.epochs}
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    run_dir = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    model = LLM(**model_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    step = 0
    epoch = 0
    best_loss = float("inf")

    def save(name):
        return save_checkpoint(
            run_dir / name, model, optimizer, model_config=model_config,
            vocabulary=sorted_text, training_config=training_config,
            dataset_sha256=digest, epoch=epoch, step=step, history=history,
        )

    print(f"Run folder: {run_dir.resolve()}", flush=True)
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            for inputs, targets in create_batches(train_data, args.batch_size, args.seq_length):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                logits = model(inputs)
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss; inspect the last periodic checkpoint.")
                loss.backward()
                optimizer.step()
                step += 1
                if step % args.save_every == 0:
                    path = save(f"step-{step:08d}.pt")
                    print(f"Epoch {epoch}, step {step}, batch loss {loss.item():.4f}; saved {path.name}", flush=True)

            # Save before validation/plotting so a reporting failure cannot lose the epoch.
            save(f"epoch-{epoch:03d}-weights.pt")
            train_metrics = evaluate(model, train_data, device, args.batch_size, args.seq_length)
            val_metrics = evaluate(model, val_data, device, args.batch_size, args.seq_length)
            row = {"epoch": epoch, "step": step}
            for split_name, metrics in (("train", train_metrics), ("val", val_metrics)):
                row.update({f"{split_name}_{key}": value for key, value in metrics.items()})
            history.append(row)
            epoch_path = save(f"epoch-{epoch:03d}.pt")
            if val_metrics["loss"] < best_loss:
                best_loss = val_metrics["loss"]
                (run_dir / "best_checkpoint.txt").write_text(epoch_path.name, encoding="utf-8")
            write_history(history, run_dir)
            try:
                plot_history(history, run_dir)
            except ImportError:
                print("Matplotlib unavailable; metrics and checkpoints are saved. Replot with validate.py after installing it.", flush=True)
            print(f"Epoch {epoch}: train loss {train_metrics['loss']:.4f}, val loss {val_metrics['loss']:.4f}", flush=True)
    except KeyboardInterrupt:
        path = save(f"interrupted-step-{step:08d}.pt")
        print(f"Training interrupted; saved {path}", flush=True)
        return
    print(f"Finished. Best checkpoint: {(run_dir / 'best_checkpoint.txt').read_text(encoding='utf-8')}", flush=True)


if __name__ == "__main__":
    main()
