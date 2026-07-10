"""generate.py — Phase 5 inference for text-to-sql-qlora-v2.

Generates SQL predictions for every row of a prepared jsonl (dev.jsonl for
the final exam), from either the base model or a LoRA adapter, and writes
them one-per-line — the exact format eval/harness.py consumes.

    Base model:      python src/generate.py --data data/dev.jsonl --out preds_base.txt
    Fine-tuned:      python src/generate.py --data data/dev.jsonl \
                         --adapter checkpoints/r16/adapter_final --out preds_ft.txt

Greedy decoding (do_sample=False): the exam answer should be the model's
single best guess, deterministic and reproducible.
"""

import argparse
import json
import re
import time
from pathlib import Path

from unsloth import FastLanguageModel  # must import before transformers
import torch

BASE_MODEL = "unsloth/Phi-3-mini-4k-instruct"
MAX_SEQ_LENGTH = 3584
MAX_NEW_TOKENS = 300


def clean_sql(text: str) -> str:
    """One prediction per output line: strip fences/labels, collapse newlines."""
    text = text.strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    if text.lower().startswith("sql:"):
        text = text[4:]
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "SELECT 1 -- empty generation"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=None,
                    help="path to adapter_final; omit for base model")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap examples (for smoke-testing this script)")
    args = ap.parse_args()

    model_name = args.adapter if args.adapter else BASE_MODEL
    print(f"[load] {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"          # required for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = []
    with open(args.data, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit is not None and i >= args.limit:
                break
            rows.append(json.loads(line))
    print(f"[data] {len(rows)} examples from {args.data}")

    prompts = [
        tokenizer.apply_chat_template(
            r["messages"][:-1],               # system + user; drop gold answer
            tokenize=False,
            add_generation_prompt=True,       # cue the assistant turn
        )
        for r in rows
    ]

    preds = []
    t0 = time.time()
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start:start + args.batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=MAX_SEQ_LENGTH).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        for seq in new_tokens:
            preds.append(clean_sql(tokenizer.decode(seq, skip_special_tokens=True)))
        done = min(start + args.batch_size, len(prompts))
        rate = done / (time.time() - t0)
        eta = (len(prompts) - done) / max(rate, 1e-9)
        print(f"[gen] {done}/{len(prompts)}  ({rate:.1f} ex/s, eta {eta/60:.0f} min)",
              flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(p + "\n")
    print(f"[done] {len(preds)} predictions -> {args.out} "
          f"in {(time.time()-t0)/60:.1f} min")

    print("[sample]", preds[0][:150])


if __name__ == "__main__":
    main()
