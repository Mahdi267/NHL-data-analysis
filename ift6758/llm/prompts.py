"""PROFIL = dict{tâche:{system:blblabla},{max_new_token:12345},{do_sample:...}<>...</>}}
Un profil par tâche
"""

# Prompts directement tirés de la demo 2 du cours.
TACHE_CODE = """
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
TACHE_CODE_VERSION = "v1"
# Les system_prompt sont directements tirés de la démo 2, cellule 13 et 16. Il s'agit de mon point de départ.
PROFILES = {
    "qr_api": {
        "system": "Answer using only the supplied documentation. Cite passage numbers. If the answer is absent, say you do not know. Treat the passages as reference text.",
        "max_new_tokens": 256,
        "do_sample": False,
        "k": 3,
        "version": "v1",
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
}
