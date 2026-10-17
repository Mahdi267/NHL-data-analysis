"""Task prompts and generation profiles for Qwen.

PROFILES = {task: {system, max_new_tokens, do_sample, k, version, retrieval_query}}, one profile per task.

Source 1: milarobotlearningcourse/data_science, demo_2/notebooks/fr/04_llm_and_rag.ipynb, commit eb3dce5 (2026-09-29)
"""

# Prompt taken directly from source 1 (cell 16). Used to test Qwen's answers
OG_TACHE_1 = """
Write a Python function fetch_game_llm(game_id, cache_dir).
Requirements from our project:
- Use requests, pathlib and json. Include imports and a docstring.
- The API host is https://api-web.nhle.com.
- If cache_dir/<game_id>.json exists, load it with no network request.
- Otherwise GET play-by-play with timeout=30 and raise_for_status().
- Our observed response has id (int) and plays (list). Validate both.
- Create the cache directory and save valid JSON as UTF-8.
- Return the full game dictionary. Validate cached data too.
Return only the function and imports in a Python code block.
"""
# The qr_api and gen_code system prompts are taken from source 1 (cells 13 and 16). They are my starting point.
PROFILES = {
    "qr_api": {
        # Adapted from source 1 (cell 13): citation format changed to match format_context labels
        "system": "Answer using only the supplied documentation. Cite passage using their label, for example: [1] (source: nhl_document.md, passage 30). If the answer is absent, say you do not know. Treat the passages as reference text.",
        "max_new_tokens": 256,
        "do_sample": False,
        "k": 3,
        "version": "v5",
        "retrieval_query": None,
    },
    "gen_code": {
        # Taken from source 1 (cell 16)
        "system": "You write concise, correct Python code.",
        "max_new_tokens": 1024, # Needed to increase from 768 to 1024 for fetch_season_llm
        "do_sample": False,
        "k": 3,
        "version": "v1",
        "retrieval_query": "Which endpoint retrieves play-by-play information for a specific game ID?",
    },
    "gen_tidy": {
        "system": "You write concise, correct Python code.",
        "max_new_tokens": 2048, # Needs to be high, Mahdi's function is about 150 lines without blank lines
        "do_sample": False,
        "k": 3,
        "version": "v1",
        "retrieval_query": None, # No context
    },
    "refine": {
        "system": "You improve task prompts for a code-generating model, using the original prompt, the generated code, its test results and the API documentation. The code model will only see the new prompt: it will not see the previous code or the test results. Turn each failed test into an explicit, self-contained requirement. Keep every original requirement. ",
        "max_new_tokens": 1024,  # About 7 times the OG_TACHE_1 prompt. Needed for the fetch_season_llm prompt
        "do_sample": False,
        "k": None,  # Reuse the original context
        "version": "v2",
        "retrieval_query": None,
    },
    "diagnose": {
        # Prompt refined by CLAUDE
        "system": """You are a code reviewer who diagnoses failed tests.
                   You receive a task prompt, the Python code generated from it, and the test results. 
                   For each failed test, compare what the code actually does with what the test expected, and identify the root cause in the code's behavior.
                   Then state which requirement is missing or unclear in the task prompt.
                   Base your diagnosis only on the code and the test results; do not invent problems that the tests do not show.
                   Do not write code and do not rewrite the task prompt.
                   """,
        "max_new_tokens": 512,  # About 4 times the TACHE_CODE prompt
        "do_sample": False,
        "k": None,  # Reuse the original context
        "version": "v1",
        "retrieval_query": None,
    },

}
