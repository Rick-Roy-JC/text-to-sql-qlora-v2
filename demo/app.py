"""Text-to-SQL demo — Phi-3-mini + QLoRA (r=16) fine-tuned on Spider.

Runs the merged, GGUF-quantized (q4_k_m) model on CPU via llama.cpp.
Prompt format matches training exactly (see src/prepare_data.py in the repo).
"""

import gradio as gr
from llama_cpp import Llama

GGUF_REPO = "this-is-rickroy/phi3-mini-spider-sql-gguf"
GGUF_FILE = "phi3-mini-spider-sql.q4_k_m.gguf"

SYSTEM_PROMPT = (
    "You are a text-to-SQL assistant. Given a database schema and a question, "
    "write a single SQLite-compatible SQL query that answers the question. "
    "Output only the SQL query."
)

print("Loading model (first boot downloads ~2.3 GB — a few minutes)...")
llm = Llama.from_pretrained(
    repo_id=GGUF_REPO,
    filename=GGUF_FILE,
    n_ctx=4096,
    n_threads=2,
    verbose=False,
)
print("Model ready.")

# --- example schemas (from Spider dev; picked because the model handles them well)
SCHEMA_CONCERT = """stadium(Stadium_ID NUMBER, Location TEXT, Name TEXT, Capacity NUMBER, Highest NUMBER, Lowest NUMBER, Average NUMBER)
  primary key: Stadium_ID
singer(Singer_ID NUMBER, Name TEXT, Country TEXT, Song_Name TEXT, Song_release_year TEXT, Age NUMBER, Is_male OTHERS)
  primary key: Singer_ID
concert(concert_ID NUMBER, concert_Name TEXT, Theme TEXT, Stadium_ID TEXT, Year TEXT)
  primary key: concert_ID
singer_in_concert(concert_ID NUMBER, Singer_ID TEXT)
  primary key: concert_ID
foreign key: concert.Stadium_ID -> stadium.Stadium_ID
foreign key: singer_in_concert.Singer_ID -> singer.Singer_ID
foreign key: singer_in_concert.concert_ID -> concert.concert_ID"""

SCHEMA_PETS = """Student(StuID NUMBER, LName TEXT, Fname TEXT, Age NUMBER, Sex TEXT, Major NUMBER, Advisor NUMBER, city_code TEXT)
  primary key: StuID
Has_Pet(StuID NUMBER, PetID NUMBER)
Pets(PetID NUMBER, PetType TEXT, pet_age NUMBER, weight NUMBER)
  primary key: PetID
foreign key: Has_Pet.StuID -> Student.StuID
foreign key: Has_Pet.PetID -> Pets.PetID"""


def generate_sql(schema: str, question: str) -> str:
    if not schema.strip() or not question.strip():
        return "-- please provide both a schema and a question"
    user_msg = f"### Database schema:\n{schema}\n\n### Question:\n{question}\n\n### SQL:"
    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n{user_msg}<|end|>\n"
        f"<|assistant|>\n"
    )
    out = llm(
        prompt,
        max_tokens=300,
        temperature=0.0,
        stop=["<|end|>", "<|user|>"],
    )
    return out["choices"][0]["text"].strip()


with gr.Blocks(title="Text-to-SQL — Phi-3-mini QLoRA") as demo:
    gr.Markdown(
        """# Text-to-SQL — Phi-3-mini fine-tuned on Spider (QLoRA, r=16)
Paste a database schema, ask a question in English, get SQL.
**69.4% execution accuracy** on Spider dev (base model: 59.0%).
[Model & training code](https://github.com/Rick-Roy-JC/text-to-sql-qlora-v2) ·
running 4-bit GGUF on free CPU — generation takes a few seconds."""
    )
    with gr.Row():
        with gr.Column():
            schema_box = gr.Textbox(label="Database schema", lines=12,
                                    value=SCHEMA_CONCERT)
            question_box = gr.Textbox(
                label="Question",
                value="How many singers are from each country?")
            btn = gr.Button("Generate SQL", variant="primary")
        with gr.Column():
            sql_out = gr.Code(label="Generated SQL", language="sql")

    btn.click(generate_sql, inputs=[schema_box, question_box], outputs=sql_out)

    gr.Examples(
        examples=[
            [SCHEMA_CONCERT, "How many singers are from each country?"],
            [SCHEMA_CONCERT,
             "Show the stadium name and the number of concerts in each stadium."],
            [SCHEMA_CONCERT,
             "What are the names of singers who performed in a concert in 2014?"],
            [SCHEMA_PETS, "How many students have a dog?"],
            [SCHEMA_PETS,
             "Find the average age of students who do not have any pets."],
        ],
        inputs=[schema_box, question_box],
    )

demo.launch()
