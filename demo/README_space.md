---
title: Text-to-SQL Phi-3 QLoRA
emoji: 🗄️
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Text-to-SQL — Phi-3-mini + QLoRA (Spider)

Natural-language questions -> SQLite queries. Phi-3-mini-4k-instruct fine-tuned
with QLoRA (r=16, Unsloth) on the Spider benchmark.

**Spider dev (1,034 held-out examples, greedy decoding):**
execution accuracy 69.4% (base: 59.0%), exact match 47.3% (base: 11.5%).

Served as a q4_k_m GGUF via llama.cpp on free CPU hardware.
Training code, ablations, and lessons: https://github.com/Rick-Roy-JC/text-to-sql-qlora-v2
