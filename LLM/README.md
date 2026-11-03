# Tiny character-level language model

A learning project implementing a single decoder Transformer block with causal attention and RoPE in PyTorch. Dataset: Tiny Shakespeare, stored in `data/input.txt`.

The workspace `AGENTS.md` defines the mentoring and review rules.

## Files

- `tokeniser.py`: character/ID mappings, contiguous 80/10/10 splits, and shifted batches.
- `llm.py`: model and training entry point.
- `checkpoint.py`: save and reload `state_dict` checkpoints.
- `validate.py`: evaluate a checkpoint and export metrics and learning curves.

## Run from the workspace root

Requires PyTorch. Graphs additionally require Matplotlib; these changes do not install packages.

```powershell
python LLM/llm.py --epochs 1
```

Training does not start when you import the model. Launching the command above does start a full epoch; it is not a quick check. The default model settings remain 512 embedding dimensions, 8 heads, batch size 4, sequence length 8, and Adam learning rate 0.01. These are existing exploratory settings, not tuned results. The default epoch count without `--epochs` is 20.

Each invocation creates a unique `LLM/runs/<run-id>/` folder. The training loop visits all training batches once per epoch.

## What is saved

- `step-XXXXXXXX.pt`: periodic checkpoints, every 5,000 optimizer updates by default (`--save-every` changes the interval).
- `epoch-NNN-weights.pt`: weights and optimizer saved immediately after the training epoch, before evaluation.
- `epoch-NNN.pt`: the same model plus updated training/validation metric history.
- `best_checkpoint.txt`: filename of the epoch checkpoint with the lowest validation loss so far.
- `history.json` and `history.csv`: evaluation metrics after each epoch.
- `learning_curves.png`: training/validation cross-entropy, perplexity, and next-character accuracy, if Matplotlib is available.
- `interrupted-step-XXXXXXXX.pt`: checkpoint saved when Ctrl+C interrupts training normally. Forced process termination or power loss cannot trigger this save; periodic checkpoints provide recovery artifacts in that case.

Checkpoint files contain `model_state_dict`, `optimizer_state_dict`, architecture settings, the ordered character vocabulary, training settings, dataset SHA-256, epoch, step, and history. They are created without overwriting existing checkpoint files. Optimizer state is retained for future resume support; automatic training resume is not implemented.

The earlier training run did not save its weights. These files support future runs; they cannot recover that model.

## Validate saved weights

Replace `<run-id>` with the printed run folder and choose an existing checkpoint:

```powershell
python LLM/validate.py LLM/runs/<run-id>/epoch-001.pt
```

Validation reloads the architecture and weights, checks the dataset fingerprint, and uses the saved vocabulary, split ratios, and batch settings. It disables dropout and gradient tracking. The report includes the number of evaluated target tokens, token-weighted mean cross-entropy, perplexity, and next-character accuracy. Perplexity is `exp(cross_entropy)`; lower is better. Accuracy is a fraction between 0 and 1.

Results go into a folder named after the checkpoint beside it. Use `--output-dir` to choose another folder or `--device cuda` to evaluate on a GPU. The saved history is used to redraw the graphs. A periodic checkpoint from before the first epoch's evaluation has no history yet, so it can be evaluated but cannot produce a learning curve.

Training and validation curves are both measured after the epoch in evaluation mode, making them comparable. The existing batch generator discards incomplete batches and trailing windows; metrics describe the evaluated full batches, not every character in the split.

Keep the test split for a final evaluation after selecting settings with validation:

```powershell
python LLM/validate.py LLM/runs/<run-id>/epoch-001.pt --split test
```

Save metric reports and selected plots for the project write-up. Generated runs and dataset files are excluded by the local `.gitignore`; copy only the reports/figures you want to share into a separate tracked folder. No GitHub commit or push is performed by these scripts.

Dataset source: [Karpathy's Tiny Shakespeare corpus](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt).

## Prepare the dataset after cloning

From the repository root in PowerShell, install the dependencies and download Tiny Shakespeare before importing or running the tokenizer/model:

```powershell
python -m pip install -r LLM/requirements.txt
New-Item -ItemType Directory -Path LLM/data -Force | Out-Null
Invoke-WebRequest -Uri https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -OutFile LLM/data/input.txt
```

The checked-in result files describe completed local experiments. Checkpoints and data are excluded from Git. To generate new checkpoints and graphs, run the explicit training commands in the repository README.
