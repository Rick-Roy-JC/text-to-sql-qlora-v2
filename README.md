# Text-to-SQL with QLoRA — Phi-3-mini on Spider

Fine-tuning **Phi-3-mini-4k-instruct** for text-to-SQL on the **Spider** benchmark
using QLoRA (4-bit base, LoRA adapters) with an **Unsloth** training stack, a clean
LoRA-rank ablation, and evaluation by **execution accuracy** against real sqlite
databases.

**[🔴 Live demo](https://rick-roy-jc--text-to-sql-phi3-spider-web.modal.run)** ·
[Adapter on HF Hub](https://huggingface.co/this-is-rickroy/phi3-mini-spider-qlora-r16) ·
[GGUF (q4_k_m)](https://huggingface.co/this-is-rickroy/phi3-mini-spider-sql-gguf)

## Results

Spider dev set — 1,034 held-out examples, greedy decoding, never used for
training or model selection:

| Model | Exact Match | Execution Accuracy |
|---|---|---|
| Phi-3-mini-4k-instruct (base) | 11.51% | 58.99% |
| **+ QLoRA fine-tune (r=16)** | **47.29%** | **69.44%** |

Fine-tuning removes ~25% of the base model's execution errors and quadruples
exact-match agreement with Spider's SQL style. The gap between the two metrics
is itself informative: the base model often writes *correct* SQL in its own
dialect (59% executes correctly despite 11.5% string match), so string-level
metrics alone would drastically understate it — which is why execution
accuracy, computed by running gold and predicted SQL against each example's
sqlite database, is the headline metric here.

## LoRA rank ablation

2,500-example subset, 2 epochs, alpha=32 and dropout=0.05 fixed — only `r`
varies:

| LoRA rank | Final eval loss |
|---|---|
| r=8 | 0.0582 |
| **r=16** | **0.0569** |
| r=32 | 0.0570 |

Capacity gains saturate at r=16; doubling the adapter to r=32 bought nothing.
The final model trains r=16 on the full ~6.3k training split for 3 epochs
(final eval loss 0.0435).

## Design decisions

**Spider over WikiSQL.** Spider's multi-table schemas with JOINs make the
schema-in-prompt formulation meaningful; WikiSQL is single-table. A token
audit of the serialized schemas (max 3,395 tokens) set `max_seq_length=3584`
— verified against the data before any training config was written.

**Phi-3-mini over Llama-3.2-3B.** MIT license (no gated-repo wait) and a
stronger code/SQL baseline.

**Unsloth over raw HF Trainer.** ~2x faster on T4, lower VRAM, and it selects
precision from the GPU's actual capability — which makes an entire class of
silent-slowdown bugs (bf16 on a Turing card) structurally impossible. The full
r=8/16/32 ablation ran in a single overnight Kaggle session, r=8 and r=16 in
parallel on separate T4s.

**Eval harness first.** The exact-match + sqlite execution-accuracy harness
was built and unit-tested (including read-only enforcement against
model-generated DML, and order-sensitivity only when the gold query orders)
*before* any training — then validated at 1.0000/1.0000 by scoring gold
queries as their own predictions. The payoff metric existed before a single
GPU-hour was spent.

**Loss masked to assistant tokens.** The model is trained only on predicting
SQL, not on regurgitating the schema. (Side effect documented in
[LESSONS.md](LESSONS.md): eval losses are only comparable under identical
masking schemes.)

**Held-out discipline.** `dev.json` was untouched until the single final
evaluation; train/val is a seeded 90/10 split of `train_spider.json`.

## Error analysis

The fine-tuned model's invalid-SQL rate ticks slightly *up* versus base
(8.7% vs 8.3%): it attempts more ambitious multi-join queries and
occasionally fumbles them, where the base model more often writes simple,
valid, wrong SQL. Per-example reports for both models are in
`eval/results/` for inspection.

## Deployment

The adapter is merged into the base weights, quantized to **GGUF q4_k_m**
(~2.3 GB), and served with llama.cpp on **Modal** serverless CPU — the demo
scales to zero when idle and costs nothing to keep alive. `demo/app.py` is
the same app for local use or HF Spaces.

## Repository map

```
src/prepare_data.py    Spider -> prompts with serialized schemas, seeded splits
src/token_audit.py     verifies max_seq_length against the actual data
src/train.py           Unsloth QLoRA training (smoke test / ablation / final)
src/generate.py        batched greedy inference, base or adapter
eval/harness.py        exact match + sqlite execution accuracy
tests/test_harness.py  unit tests for the harness (synthetic database)
notebooks/             one Kaggle notebook per phase, in order
demo/                  Gradio app (local / Spaces) + Modal deployment
LESSONS.md             every bug from v1 and how v2 prevents it structurally
```

## Reproduce

```bash
pip install -r requirements.txt
python src/prepare_data.py --spider_dir /path/to/spider --out_dir data
python src/token_audit.py --data_dir data --max_seq_length 3584
python tests/test_harness.py
# then notebooks/phase2..phase5 on Kaggle (T4), in order
```

---

*This is a v2. The first attempt died to a cursed repo name, a silent 60x
bf16-on-T4 slowdown, and Colab quota walls — see [LESSONS.md](LESSONS.md)
for the full casualty report and what each failure taught.*
