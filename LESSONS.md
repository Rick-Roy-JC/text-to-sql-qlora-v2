# Lessons Carried Forward from v1

This project is a v2 restart. The first attempt trained a successful r=8 adapter
(eval_loss 0.1387) but died to scaffolding problems: a cursed repo name, Colab
free-tier quota exhaustion mid-ablation, and a training stack that made silent
performance bugs possible. Every bug below was hit (or caught just in time) in v1
and is structurally prevented in v2.

## Bugs caught before they burned GPU time

### 1. Stale `max_seq_length` placeholder (768)
A Phase 2 token audit found with-schema Spider examples up to **3,395 tokens**.
At 768, most of the schema would have been silently truncated — defeating the
entire point of choosing Spider (multi-table JOINs) over WikiSQL.
**v2 rule:** `token_audit.py` runs before any training config is written.
`MAX_SEQ_LENGTH = 3584`, verified against the actual data, not assumed.

### 2. Unset `per_device_eval_batch_size`
HF Trainer defaults eval batch size to 8. At 3,584-token sequences that blows
past 16 GB on a T4, and the crash hit *before* the checkpoint save — losing a
1h40m run. **v2 rule:** eval batch size is set explicitly to 1 on day one, and
the smoke test (Phase 2) deliberately runs an eval step before any long run.

### 3. `resume_from_checkpoint=True` on first run
Errors when no checkpoint exists. **v2 rule:** guarded with
`get_last_checkpoint()` (or Unsloth equivalent) from the first commit.

### 4. Phi-3 `pad_token_id == eos_token_id`
Unmasked padding would train the model to treat padding as a valid stop signal.
**v2 rule:** padded positions masked with `labels = -100`; the smoke test
prints a decoded sample with its label mask for eyeball verification.

## Bugs discovered live (burned time)

### 5. bf16 on a T4 — the ~60x silent slowdown
T4 is Turing (sm_75): fp16 tensor cores, **no bf16 support**. bf16 silently
falls back to a non-tensor-core path — 113 sec/step instead of ~8–15. No error,
no warning, just a run that would take weeks.
**v2 rule:** Unsloth auto-detects GPU capability and selects precision, making
this bug structurally impossible. The smoke test still asserts sec/step is
single-digit before any long run. Also: on Kaggle, prefer **T4x2 over P100** —
P100 lacks tensor cores entirely and quietly recreates a milder version of this.

### 6. Leading hyphen in repo name (`-text-to-sql-qlora`)
`%cd -text-to-sql-qlora` was parsed by IPython as a flag (`option -t not
recognized`), so the working directory never changed, so every subsequent
`!python src/train.py` failed with `No such file or directory`. One character,
three cascading errors, one full debug session.
**v2 rule:** this repo is named `text-to-sql-qlora-v2`. No leading punctuation.
Ever.

### 7. Python variables don't propagate to shell cells
`CHECKPOINT_ROOT` set as a Python variable, referenced as `$CHECKPOINT_ROOT` in
a `!python` cell — shell cells don't inherit the Python namespace.
**v2 rule:** anything a shell cell needs goes through
`os.environ['VAR'] = value`, set in one dedicated config cell at the top of
every notebook.

### 8. `scaler.pt` / paged-optimizer incompatibility on resume
A checkpoint saved during a run that had a GradScaler includes `scaler.pt`;
resuming with `paged_adamw_8bit` + recent accelerate can initialize no scaler,
and Trainer crashes loading `scaler.pt` into `None`.
**v2 rule:** `train.py` deletes a stale `scaler.pt` before resume if present
(guard carried over verbatim from v1), and Phase 2 explicitly tests
kill-and-resume before the ablation.

## Architectural decisions that paid off (kept in v2)

- **Checkpoint persistence + resume guard:** every v1 crash cost at most a few
  minutes of compute. v2 keeps this, simplified: Kaggle's 20 GB persistent
  storage + background execution replaces the Colab/Drive rclone dance.
- **Clean ablation design:** only `r` varies (8/16/32); `alpha=32`,
  `dropout=0.05` fixed. Kept unchanged so v2 results remain comparable to the
  completed v1 r=8 run.
- **dev.json as untouched held-out eval**, 90/10 train/val from
  train_spider.json. Kept unchanged.

## v2 upgrades (not in v1)

- **Unsloth** replaces raw HF Trainer + BitsAndBytes: ~2x faster, lower VRAM,
  correct precision selection by construction. (Free tier is single-GPU — on
  Kaggle T4x2 the second GPU idles. Expected; still faster than v1's stack.)
- **Eval harness built first** (Phase 1, CPU-only): exact match + sqlite
  execution accuracy exist and are unit-tested before a single GPU-hour is
  spent. v1 inverted this and the payoff metric never materialized.
- **Kaggle as primary compute** (30 GPU-hrs/week, predictable), Lightning AI
  academic credits as backup fuel.
