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
        "system": "You take existing AI model prompts and make them better adn more efficient using the information available to you. The prompts will be used by a model that generates code. You need to help the model identify mistakes in its previous code to help him achieve the original task. Keep all the original requirements from the previous prompt and add the new one to help the model with its mistakes in the previous code.",
        "max_new_tokens": 256, # Environ 2 fois plus que le prompt TÂCHE_CODE
        "do_sample": False,
        "k": None, # On réutilise contexte original
        "version": "v1",
        "retrieval_query": None,
    },
}
