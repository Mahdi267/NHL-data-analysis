"""
Chargement de Qwen3-4B-Instruct-2507 et fonction d'appel simple.                                                                                                                                                                                                                                                                  │tutoriels/test.ipynb
                                                                                                                                                                                                                                                                                                                                     │
Code adapté de la démo 2 du cours IFT 6758 (cellule « Charger Qwen ») :                                                                                                                                                                                                                                                              │────────────────────────────────────────────────────────────────────────────────────────
    demo_2/notebooks/fr/07_llm_and_rag_FP16.ipynb                                                                                                                                                                                                                                                                                    │tutoriels/Conversation_IA.md (untracked)
    Dépôt : https://github.com/milarobotlearningcourse/data_science                                                                                                                                                                                                                                                                  │────────────────────────────────────────────────────────────────────────────────────────
    Version consultée : commit eb3dce5 (2026-09-29)                                                                                                                                                                                                                                                                                  │New file not yet staged.
                                                                                                                                                                                                                                                                                                                                     │Run `git add :/tutoriels/Conversation_IA.md` to see line counts.
Modifications : chargement déplacé dans load_qwen() pour ne pas charger                                                                                                                                                                                                                                                              │
le modèle à l'import ; gen_pipe et tokenizer passés en paramètres à ask_qwen().                                                                                                                                                                                                                                                      │────────────────────────────────────────────────────────────────────────────────────────
"""

import sys

if sys.version_info[:2] != (3, 11):
    raise RuntimeError("Use a Python 3.11 kernel: the course .venv locally, or a compatible Colab runtime.")

import numpy as np
import pandas as pd
import torch
from importlib.metadata import version
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline




# Adapté de la démo 2 (04_llm_and_rag.ipynb, cellule « Vérification »).                                                                                                                                                                                                                                                         │
def check_env():
    print("Python:", sys.version.split()[0])
    print("NumPy:", np.__version__, "| pandas:", pd.__version__)
    print("Transformers:", version("transformers"))
    print("Sentence Transformers:", version("sentence-transformers"))
    print("CUDA GPU:",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unavailable — CPU/Apple GPU fallback")

# Adapté de la démo 2 (04_llm_and_rag.ipynb, cellule « Charger Qwen »).
def load_qwen(device=None, model_id="Qwen/Qwen3-4B-Instruct-2507"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_options = {"dtype": torch.float32 if device == "cpu" else torch.float16}
    if device == "cuda":
        model_options["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_options)
    if device != "cuda":
        model = model.to(device)
    gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("Loaded", model_id, "on", device)
    return gen_pipe, model, tokenizer

# Adapté de la démo 2 (04_llm_and_rag.ipynb, cellule « Charger Qwen »).
def ask_qwen(gen_pipe, tokenizer,prompt, system="You are a helpful assistant.", max_new_tokens=512):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return gen_pipe(text, max_new_tokens=max_new_tokens, do_sample=False,
                        return_full_text=False, pad_token_id=tokenizer.eos_token_id)[0]["generated_text"].strip()
