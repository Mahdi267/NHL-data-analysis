"""
Ensemble de fonctions d'appel simple pour gérer le modèle. Configuré pour Qwen3-4B-Instruct-2507.

Code en partie dérivé de la démo 2 du cours IFT 6758.

METTRE UNE LISTE DE REFERENCE DE LA DEMO 2

"""
import sys

import prompt_toolkit

from ift6758.llm.qwen_utils_2 import DEFAULT_SYSTEM

if sys.version_info[:2] != (3, 11):
    raise RuntimeError("Use a Python 3.11 kernel: the course .venv locally, or a compatible Colab runtime.")

import time
import gc
import numpy as np
import pandas as pd
import torch
from importlib.metadata import version
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

def pick_device():
    """Retourne "cuda", "mps" ou "cpu" """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

# Adapté de la démo 2 (04_llm_and_rag.ipynb, cellule « Vérification »).                                                                                                                                                                                                                                                         │
def check_env():
    print("Python:", sys.version.split()[0])
    print("NumPy:", np.__version__, "| pandas:", pd.__version__)
    print("Transformers:", version("transformers"))
    print("Sentence Transformers:", version("sentence-transformers"))
    print("CUDA GPU:",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unavailable — CPU/Apple GPU fallback")

# Adapté de la démo 2 (04_llm_and_rag.ipynb, cellule « Charger Qwen »).
def load_qwen(device=None, model_id="Qwen/Qwen3-4B-Instruct-2507"):
    if device == None:
        device = pick_device()

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

def generate(gen_pipe, tokenizer, messages, max_new_tokens=512, do_sample = False):
    """Recoit une liste de messages déjà construite. Retourne {"text","n_token_in", "n_token_out", "duration_s"} """
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    n_token_in = len(tokenizer(text)["input_ids"])
    t0 = time.perf_counter()
    out = gen_pipe(
        text,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )[0]["generated_text"].strip()
    duration_s = time.perf_counter()-t0
    n_token_out = len(tokenizer(out)["input_ids"])
    return {"text":text, "n_token_in":n_token_in, "n_token_out":n_token_out, "duration_s":duration_s}



def free_qwen(device):
    """Fonction qui permet de mettre qwen en pause pour libérer du compute power."""
    gc.collect()
    torch.cuda.empty_cache() if device == "cuda" else (torch.mps.empty_cache() if device == "mps" else None)

def build_messages(prompt,context,system="You are a helpful assistant."):
    """Construire message système et message user et ajouter context s'il y a lieu"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    return messages

