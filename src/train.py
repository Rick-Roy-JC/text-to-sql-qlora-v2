"""train.py — text-to-sql-qlora-v2. Unsloth QLoRA fine-tuning of Phi-3-mini.

One script for all training phases:

    Smoke test (Phase 2):  python src/train.py --smoke
    Ablation    (Phase 3): python src/train.py --rank 8  --subset 2500 --epochs 2
    Final run   (Phase 4): python src/train.py --rank <winner> --epochs 3

Carries forward every relevant v1 fix (see LESSONS.md):
  #2 eval batch size explicitly 1
  #3 resume guarded with get_last_checkpoint()
  #4 loss computed on assistant tokens only (prompt/padding masked)
  #5 precision auto-selected by capability (fp16 on T4) — never hardcoded
  #8 stale scaler.pt deleted before resume
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from unsloth import FastLanguageModel, is_bfloat16_supported  # must import before transformers
from datasets import Dataset
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only

MODEL_NAME = "unsloth/Phi-3-mini-4k-instruct"  # Unsloth mirror of microsoft/Phi-3-mini-4k-instruct
MAX_SEQ_LENGTH = 3584          # verified by src/token_audit.py — do not lower
LORA_ALPHA = 32                # fixed across ablation (only r varies)
LORA_DROPOUT = 0.05            # kept from v1 for comparability with the r=8 run.
                               # NOTE: nonzero dropout disables some Unsloth fast
                               # paths (it will print a warning) — accepted cost.
SEED = 42
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def load_jsonl_as_text(path: Path, tokenizer, limit: int | None = None) -> Dataset:
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            msgs = json.loads(line)["messages"]
            rows.append({"text": tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False)})
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--output_root", default=os.environ.get("CHECKPOINT_ROOT", "checkpoints"))
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--subset", type=int, default=None,
                    help="cap training examples (2500 for ablation; omit for full)")
    ap.add_argument("--max_steps", type=int, default=-1,
                    help="hard step cap; overrides epochs when > 0")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--save_steps", type=int, default=50)
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--smoke", action="store_true",
                    help="Phase 2 mode: tiny subset, 20 steps, eval+save at 10, timing asserts")
    args = ap.parse_args()

    if args.smoke:
        args.subset = 50
        args.max_steps = 20
        args.save_steps = 10
        args.eval_steps = 10
        args.grad_accum = 2

    run_name = f"r{args.rank}" + ("-smoke" if args.smoke else "")
    output_dir = Path(args.output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- v1 fix #5: precision by capability, never hardcoded ----------------
    bf16 = is_bfloat16_supported()
    print(f"[precision] bf16 supported: {bf16} -> using {'bf16' if bf16 else 'fp16'}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,          # None = Unsloth auto-detects (fp16 on T4)
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    data_dir = Path(args.data_dir)
    train_ds = load_jsonl_as_text(data_dir / "train.jsonl", tokenizer, args.subset)
    val_limit = 20 if args.smoke else 200   # capped val for cheap periodic eval
    val_ds = load_jsonl_as_text(data_dir / "val.jsonl", tokenizer, val_limit)
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}  run={run_name}")

    config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,          # v1 fix #2 — explicit, day one
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="adamw_8bit",                    # not paged_ -> avoids v1 bug #8 class
        fp16=not bf16,
        bf16=bf16,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        seed=SEED,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=config,
    )

    # ---- v1 fix #4: loss on assistant tokens only ---------------------------
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|user|>",
        response_part="<|assistant|>",
    )

    # eyeball check: decoded labels should show ONLY the SQL, prompt masked out
    sample_labels = [t for t in trainer.train_dataset[0]["labels"] if t != -100]
    print("[mask check] loss is computed on ->",
          repr(tokenizer.decode(sample_labels)[:120]))

    # ---- v1 fixes #3 + #8: guarded resume, stale scaler.pt removed ----------
    last_ckpt = get_last_checkpoint(str(output_dir))
    if last_ckpt:
        stale_scaler = Path(last_ckpt) / "scaler.pt"
        if stale_scaler.exists():
            print(f"[resume] deleting stale {stale_scaler}")
            stale_scaler.unlink()
        print(f"[resume] resuming from {last_ckpt}")
    else:
        print("[resume] no checkpoint found — fresh start")

    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=last_ckpt)
    elapsed = time.time() - t0

    steps_done = trainer.state.global_step - (
        int(Path(last_ckpt).name.split("-")[-1]) if last_ckpt else 0)
    sec_per_step = elapsed / max(steps_done, 1)
    print(f"[timing] {steps_done} steps in {elapsed:.0f}s -> {sec_per_step:.1f} sec/step")

    trainer.save_model(str(output_dir / "adapter_final"))
    print(f"[save] adapter -> {output_dir / 'adapter_final'}")

    # ---- Phase 2 gate assertions --------------------------------------------
    if args.smoke:
        checks = {
            "sec/step sane (<20, i.e. no v1-style 113s slowdown)": sec_per_step < 20,
            "eval ran without OOM": any("eval_loss" in h for h in trainer.state.log_history),
           "checkpoint written": any(output_dir.glob("checkpoint-*")),
        }
        print("\n=== SMOKE TEST RESULTS ===")
        for name, ok in checks.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not all(checks.values()):
            raise SystemExit("SMOKE TEST FAILED — do not proceed to Phase 3.")
        print("Checks 1-3 PASSED. Check 4 (kill+resume) is the next notebook cell.")


if __name__ == "__main__":
    main()
