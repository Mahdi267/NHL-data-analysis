"""
Set of simple helper functions to load, query and free the model. Configured for Qwen3-4B-Instruct-2507.

Code partly derived from demo 2 of the IFT 6758 course and from the Hugging Face website.

Source 1: milarobotlearningcourse/data_science, demo_2/notebooks/fr/04_llm_and_rag.ipynb, commit eb3dce5 (2026-09-29)
Source 2: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507, Qwen3-4B-Instruct-2507 Quickstart
Source 3: https://huggingface.co/learn/cookbook/advanced_rag, Hugging Face cookbook (Advanced RAG)
"""

# Adapted from source 1 (cell 4: Python version check and imports)
import sys

if sys.version_info[:2] != (3, 11):
    raise RuntimeError(
        "Use a Python 3.11 kernel: the course .venv locally, or a compatible Colab runtime."
    )
import time
import gc
import numpy as np
import pandas as pd
import torch
from importlib.metadata import version
from transformers import AutoModelForCausalLM, AutoTokenizer
from ift6758.llm.qwen_log import create_log, log_call


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
# Default system prompt taken from source 1 (ask_qwen, cell 6)
DEFAULT_SYSTEM = "You are a helpful assistant."


def pick_device():
    """Return the available compute unit (cuda > mps > cpu).

    :return: "cuda", "mps" or "cpu"
    :rtype: str
    """
    # Adapted from source 1 (cell 6), rewritten as an if/elif/else
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def check_env():
    """Print the library versions and the GPU in use.

    :return: None
    """
    # Taken from source 1 (cell 4)
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


def load_qwen(model_id=MODEL_ID, device=None):
    """Load the model and tokenizer (float16 on GPU, float32 on CPU).

    :param model_id: Hugging Face model id
    :type model_id: str
    :param device: Compute unit, picked automatically if None
    :type device: str or None
    :return: model, tokenizer, device
    :rtype: tuple
    """
    # Adapted from source 1 (cell 6) and source 2 (Quickstart)
    device = pick_device() if device is None else device
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Same dtype choice as demo 2, for compatibility across team machines
    model_options = {"dtype": torch.float32 if device == "cpu" else torch.float16}
    if device == "cuda":
        model_options["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, **model_options
    )  # Options added for compatibility
    # On CUDA the model is already placed by device_map, otherwise move it
    if device != "cuda":
        model = model.to(device)
    # Demo 2 code stops here: no transformers pipeline, so generate() can count the exact number of tokens for the log
    return model, tokenizer, device


def build_messages(prompt, context=None, system=None):
    """Build the system + user messages, with the optional context added before the prompt.

    :param prompt: User message
    :type prompt: str
    :param context: RAG context
    :type context: str or None
    :param system: System prompt, DEFAULT_SYSTEM if None
    :type system: str or None
    :return: Chat messages
    :rtype: list[dict]
    """
    system = system or DEFAULT_SYSTEM  # If system is empty, use the default
    # Prompt structure and instructions adapted from source 3
    question = (
        "".join(
            f"<context>{context}</context>\n"
            + f"Now here is the question you need to answer.\n<question>{prompt}</question>"
        )
        if context
        else prompt
    )
    # Message format taken from source 1 (ask_qwen, cell 6)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    return messages


def generate(model, tokenizer, messages, max_new_tokens=512, do_sample=False):
    """Generate the answer and count the tokens in and out.

    :param model: Loaded model
    :param tokenizer: Model tokenizer
    :param messages: Chat messages
    :type messages: list[dict]
    :param max_new_tokens: Max tokens to generate
    :type max_new_tokens: int
    :param do_sample: False for reproducibility (greedy)
    :type do_sample: bool
    :return: prompt, answer, token counts, duration and parameters
    :rtype: dict
    """
    # Adapted from source 2 (Quickstart)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

    n_token_in = model_inputs["input_ids"].shape[
        1
    ]  # Count the input tokens (input_ids has shape (1, n))
    t0 = time.perf_counter()  # Timer for the model's generation time
    # Adapted from source 1 (cell 6, generation parameters) and source 2 (Quickstart)
    generated = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )  # generated has shape (1, n_token_in + n_token_out)
    out = generated[0][n_token_in:]  # out has shape (n_token_out,)
    duration_s = time.perf_counter() - t0
    n_token_out = len(out)
    content = tokenizer.decode(out, skip_special_tokens=True).strip()
    return {
        "prompt": prompt_text,
        "answer": content,
        "n_token_in": n_token_in,
        "n_token_out": n_token_out,
        "duration_s": duration_s,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }


def free_qwen(device):
    """Empty the device cache. Call after ``del model, tokenizer``.

    :param device: Compute unit
    :type device: str
    :return: None
    """
    gc.collect()
    torch.cuda.empty_cache() if device == "cuda" else (
        torch.mps.empty_cache() if device == "mps" else None
    )

def ask_qwen(model, tokenizer, log_path, prompt, context, profile, extra=None):
    """Send a prompt to Qwen with a profile, log the call and print the answer.

    :param model: Loaded model
    :param tokenizer: Model tokenizer
    :param log_path: JSONL log file
    :type log_path: Path or str
    :param prompt: User message
    :type prompt: str
    :param context: RAG context
    :type context: str or None
    :param profile: Entry of PROFILES
    :type profile: dict
    :param extra: Extra info to log
    :type extra: dict or None
    :return: Output of generate()
    :rtype: dict
    """
    p = profile
    messages = build_messages(prompt, context, system=p["system"])
    # Generate answer
    result = generate(
        model,
        tokenizer,
        messages,
        max_new_tokens=p["max_new_tokens"],
        do_sample=p["do_sample"],
    )
    # logs results
    log_call(
        log_path,
        messages,
        result,
        extra={
            "profile": None,
            "profile_version": p["version"],
            "max_new_tokens": p["max_new_tokens"],
            "do_sample": p["do_sample"],
            "context_format": "v5", # Need to implement a incrementation method
            **(extra or {}),
        },
    )
    print(result["answer"])
    print(
        f"\n{result['n_token_in']} tokens in, {result['n_token_out']} generated, {result['duration_s']:.1f} s"
    )
    # Check if we reach max_new_tokens
    if result["n_token_out"] >= p["max_new_tokens"]:
        print("max_new_tokens reached: truncated output")
    return result
