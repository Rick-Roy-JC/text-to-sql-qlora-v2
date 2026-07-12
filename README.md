# text-to-sql-qlora-v2

QLoRA fine-tuning of **Phi-3-mini-4k-instruct** on the **Spider** text-to-SQL
benchmark, with a clean LoRA rank ablation (r=8/16/32) and evaluation by
**exact match + execution accuracy** on Spider's held-out dev set.

v2 of an earlier attempt — see [LESSONS.md](LESSONS.md) for the eight
documented bugs from v1 and how this version prevents each one structurally.
Headline upgrades: **Unsloth** training stack (~2x faster, precision-safe on
T4), and an **eval-harness-first** build order so the payoff metric exists
before any GPU time is spent.

## Results

*(Phase 5 )*

| Model | Exact Match | Execution Accuracy |
|---|---|---|
| Phi-3-mini (base) | 11.51% | 58.99% |
| + QLoRA (winning r) | 47.29% | 69.44% |

 Evaluated on Spider's held-out dev set (1,034 examples), greedy decoding, sqlite execution-accuracy harness. Fine-tuning cuts execution errors by ~25% and quadruples exact-match agreement with Spider's SQL style.
 
 | LoRA rank | Final eval loss | 
|---|---|
| r=8  | 0.0582 | 
| r=16 | 0.0569 | 
| r=32 | 0.0570 | 
 r=16 selected; doubling to r=32 bought no improvement. Final run: full training set, 3 epochs, eval loss 0.0435.


## Design decisions

- **Phi-3-mini over Llama-3.2-3B:** MIT license (no gated-repo wait), stronger
  baseline code/SQL performance. Kept from v1 so the completed v1 r=8 run
  (eval_loss 0.1387) remains directly comparable.
- **Spider over WikiSQL:** multi-table JOINs make schema-in-prompt meaningful.
- **dev.json is held out** — never used for training or model selection.
  Train/val is a seeded 90/10 split of train_spider.json.
- **Ablation:** only `r` varies (8/16/32); `alpha=32`, `dropout=0.05` fixed.
  2,500-example subset, 2 epochs. Final run: full ~6,300 examples, 3 epochs
  at the winning rank.
- **Compute:** Kaggle T4x2 (Unsloth free tier is single-GPU; one T4 trains,
  and that still beats v1's dual-availability raw-Trainer stack). P100
  deliberately avoided — no tensor cores.

## Phases

| Phase | What | Where | Status |
|---|---|---|---|
| 0 | Repo skeleton, data prep, token audit | CPU | in progress |
| 1 | Eval harness (exact match + sqlite execution), unit-tested | CPU | — |
| 2 | Unsloth smoke test: speed, eval-OOM, checkpoint, kill+resume | Kaggle T4 (~1 hr) | — |
| 3 | Rank ablation r=8/16/32 | Kaggle T4 | — |
| 4 | Final run, winning rank | Kaggle T4 | — |
| 5 | Base vs fine-tuned on dev.json | Kaggle T4 | — |
| 6 | Gradio demo on HF Spaces + writeup | — | — |

## Setup

```bash
pip install -r requirements.txt

# Download the official Spider release, then:
python src/prepare_data.py --spider_dir /path/to/spider --out_dir data
python src/token_audit.py --data_dir data --max_seq_length 3584
```

Spider's databases (`database/` directory from the release) are needed in
Phase 1+ for execution accuracy.
