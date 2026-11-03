"""Save learned weights together with the information needed to reload them."""

from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, *, model_config, vocabulary,
                    training_config, dataset_sha256, epoch, step, history):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_config,
        "vocabulary": vocabulary,
        "training_config": training_config,
        "dataset_sha256": dataset_sha256,
        "epoch": epoch,
        "step": step,
        "history": history,
    }
    # Exclusive creation avoids silently replacing an existing checkpoint.
    with path.open("xb") as handle:
        try:
            torch.save(checkpoint, handle)
        except BaseException:
            handle.close()
            path.unlink(missing_ok=True)
            raise
    return path


def load_checkpoint(path, model_class, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = model_class(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint
