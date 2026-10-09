"""
Ensemble de fonctions d'appel simple pour gérer le modèle. Configuré pour Qwen3-4B-Instruct-2507.

Code en partie dérivé de la démo 2 du cours IFT 6758.

METTRE UNE LISTE DE REFERENCE DE LA DEMO 2

"""
import sys
if sys.version_info[:2] != (3, 11):
    raise RuntimeError("Use a Python 3.11 kernel: the course .venv locally, or a compatible Colab runtime.")
import time
import gc
import numpy as np
import pandas as pd
import torch
from importlib.metadata import version
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_SYSTEM = "You are a helpful assistant."

def pick_device():
    """Fonction qui retourne le compute unit en fonction de la configuration du système
    
    :return: Compute unit
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

# Code adapté de la démo 2 (04_llm_and_rag.ipynb, cellule « Vérification »).
def check_env():
    """Fonction qui permet d'aficher la configuration du système utilisé.

    :return: None
    """
    print("Python:", sys.version.split()[0])
    print("NumPy:", np.__version__, "| pandas:", pd.__version__)
    print("Transformers:", version("transformers"))
    print("Sentence Transformers:", version("sentence-transformers"))
    print(
        "CUDA GPU:",
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "Unavailable — CPU/Apple GPU fallback",
    )
# Code adapté de la démo 2 (04_llm_and_rag.ipynb, cellule « Charger Qwen »).
def load_qwen(model_id = MODEL_ID, device = None):
    """ Fonction qui charge le modèle et qui retourne

    :param model_id: Adresse du modèle (Voir hugging Face)
    :type model_id: str
    :param device: Compute unit (dafault to None)
    :type device: str
    :rtype:
    :return:  Modèle chargé en mémoire, tokenizer, compute unit used
    """
    device = pick_device() if device is None else device
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Je garde la version de la démo 2 pour la compatibilité
    model_options = {"dtype": torch.float32 if device == "cpu" else torch.float16}
    if device == "cuda":
        model_options["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id,**model_options) # On ajoute l'option pour la compatibilité
    # Si le modèle est sur cuda, le modèle est déjà placé, sinon on dépalce
    if device != "cuda":
        model = model.to(device)
    # Ici je coupe le code de la démo 2 car MORE DATA Baby. Je veux connaitre le nombre exact de token dans le log.
    return model, tokenizer, device

def build_messages(prompt, context=None, system = None):
    system = system or DEFAULT_SYSTEM  # Si système vide, utilise default
    # https://huggingface.co/learn/cookbook/advanced_rag pour les instructions et structure
    question = (
        "".join(
            f"<context>{context}</context>\n"
            + f"Now here is the question you need to answer.\n<question>{prompt}</question>"
        )
        if context
        else prompt
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    return messages

def generate(model,tokenizer,messages,max_new_tokens=512, do_sample=False):
    """

    :param model:
    :param tokenizer:
    :param messages:
    :param max_new_tokens:
    :param do_sample:
    :return:
    """
    # https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
    prompt_text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    model_inputs = tokenizer([prompt_text],return_tensors="pt").to(model.device)

    n_token_in = model_inputs["input_ids"].shape[1] # On compte les token en entrée (format input_id = (1,n))
    t0 = time.perf_counter() # Timer pour la durée de cogitation du modèle
    # Démo et https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
    generated = model.generate(**model_inputs,max_new_tokens=max_new_tokens,do_sample=do_sample,pad_token_id =tokenizer.eos_token_id)# Generated (1,n_token_in+n_token_out)
    out = generated[0][n_token_in:] # uut a le format (n_token_out,)
    duration_s = time.perf_counter()-t0
    n_token_out = len(out)
    content = tokenizer.decode(out,skip_special_tokens=True).strip()
    return {"prompt":prompt_text, "answer":content, "n_token_in":n_token_in, "n_token_out":n_token_out, "duration_s":duration_s}

def free_qwen(device):
    """Fonction qui vide le cache du bon device. Doit suivre un delete du model et du tokenizer sinon la mémoire ne
    se videra pas. Tant que l'objet n'est pas vide, on ne peut pas le supprimer.


    :param device:
    :type device: str
    :return:
    """
    gc.collect()
    torch.cuda.empty_cache() if device == "cuda" else (torch.mps.empty_cache() if device == "mps" else None)