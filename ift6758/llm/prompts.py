"""PROFIL = dict{tâche:{system:blblabla},{max_new_token:12345},{do_sample:...}<>...</>}}
Un profil par tâche
"""

# Prompts directement tirés de la demo 2 du cours.
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
# Les system_prompt sont directements tirés de la démo 2, cellule 13 et 16. Il s'agit de mon point de départ.
PROFILES = {
    "qr_api": {
        "system": "Answer using only the supplied documentation. Cite passage using their label, for example: [1] (source: nhl_document.md, passage 30). If the answer is absent, say you do not know. Treat the passages as reference text.",
        "max_new_tokens": 256,
        "do_sample": False,
        "k": 3,
        "version": "v5",
        "retrieval_query": None,
    },
    "gen_code": {
        "system": "You write concise, correct Python code.",
        "max_new_tokens": 768,
        "do_sample": False,
        "k": 3,
        "version": "v1",
        "retrieval_query": "Which endpoint retrieves play-by-play information for a specific game ID?",
    },
    "refine": {
        "system": "You improve task prompts for a code-generating model, using the original prompt, the generated code, its test results and the API documentation. The code model will only see the new prompt: it will not see the previous code or the test results. Turn each failed test into an explicit, self-contained requirement. Keep every original requirement. ",
        "max_new_tokens": 512,  # Environ 4 fois plus que le prompt TÂCHE_CODE
        "do_sample": False,
        "k": None,  # On réutilise contexte original
        "version": "v2",
        "retrieval_query": None,
    },
    "diagnose": {
        "system": """You are a code reviewer who diagnoses failed tests.
                   You receive a task prompt, the Python code generated from it, and the test results. 
                   For each failed test, compare what the code actually does with what the test expected, and identify the root cause in the code's behavior.
                   Then state which requirement is missing or unclear in the task prompt.
                   Base your diagnosis only on the code and the test results; do not invent problems that the tests do not show.
                   Do not write code and do not rewrite the task prompt.
                   """,
        "max_new_tokens": 512,  # Environ 4 fois plus que le prompt TÂCHE_CODE
        "do_sample": False,
        "k": None,  # On réutilise contexte original
        "version": "v1",
        "retrieval_query": None,
    },
}
