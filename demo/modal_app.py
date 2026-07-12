"""Modal deployment — text-to-SQL demo (Phi-3-mini QLoRA, GGUF on CPU).

Deploy:   modal deploy demo/modal_app.py
Dev test: modal serve  demo/modal_app.py   (temporary URL, live-reloads)

The GGUF is downloaded ONCE at image-build time and baked into the container
image, so cold starts only pay model-load (~15s), not a 2.3 GB download.
Scale-to-zero: costs nothing while nobody is using it.
"""

import modal

MODEL_REPO = "this-is-rickroy/phi3-mini-spider-sql-gguf"
MODEL_FILE = "phi3-mini-spider-sql.q4_k_m.gguf"
MODEL_DIR = "/model"

SYSTEM_PROMPT = (
    "You are a text-to-SQL assistant. Given a database schema and a question, "
    "write a single SQLite-compatible SQL query that answers the question. "
    "Output only the SQL query."
)

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


def download_model() -> None:
    """Runs at image BUILD time — bakes the GGUF into the image layer."""
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE,
                    local_dir=MODEL_DIR)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "cmake")
    .pip_install(
        "llama-cpp-python>=0.3",
        "gradio>=5,<6",
        "fastapi",
        "huggingface_hub>=0.24",
    )
    .run_function(download_model)
)

app = modal.App("text-to-sql-phi3-spider", image=image)


@app.function(
    cpu=4.0,
    memory=8192,
    scaledown_window=600,     # stay warm 10 min after last visitor
    max_containers=1,
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web():
    import gradio as gr
    from fastapi import FastAPI
    from llama_cpp import Llama

    llm = Llama(model_path=f"{MODEL_DIR}/{MODEL_FILE}",
                n_ctx=4096, n_threads=4, verbose=False)

    def generate_sql(schema: str, question: str) -> str:
        if not schema.strip() or not question.strip():
            return "-- please provide both a schema and a question"
        user_msg = (f"### Database schema:\n{schema}\n\n"
                    f"### Question:\n{question}\n\n### SQL:")
        prompt = (f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
                  f"<|user|>\n{user_msg}<|end|>\n"
                  f"<|assistant|>\n")
        out = llm(prompt, max_tokens=300, temperature=0.0,
                  stop=["<|end|>", "<|user|>"])
        return out["choices"][0]["text"].strip()

    with gr.Blocks(title="Text-to-SQL — Phi-3-mini QLoRA") as demo:
        gr.Markdown(
            """# Text-to-SQL — Phi-3-mini fine-tuned on Spider (QLoRA, r=16)
Paste a database schema, ask a question in English, get SQL.
**69.4% execution accuracy** on Spider dev (base model: 59.0%).
[Model & training code](https://github.com/Rick-Roy-JC/text-to-sql-qlora-v2) ·
4-bit GGUF on serverless CPU — first request after idle takes ~30s while the
container wakes; after that, a few seconds per query."""
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

        btn.click(generate_sql, inputs=[schema_box, question_box],
                  outputs=sql_out)

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

    fastapi_app = FastAPI()
    return gr.mount_gradio_app(fastapi_app, demo, path="/")
