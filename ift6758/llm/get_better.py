"""
Je veux que le modèle puisse recevoir le prompt d'originie, le code cérer, les erreurs (retours et messages terminal) ainis que le RAG pour pouvoir améliorer lui-même sont prompt.
Une fois l'amélioration du prompt fait, je peux regarder le prompt et confirmer ou valider avant que qwen regénère un nouveau code. Je valide le code et on relance les tests.
Tous les prompts sont sauvegardé dans log comme demandé. Je peux réutiliser des éléments de chat.py
"""

def extract_code():
    """Sortir le code du bloc ```python``` de la réponse actuelle de qwem

    """
    pass

def load_function(code,name):
    """transformer str venant de extract en fonction python

    :param code:
    :param name:
    :return:
    """
    pass

def format_test_report(result):
    """rendre run_test lisible pour qwen dans son contexte

    :param result:
    :return:
    """
    pass

def build_refine_prompt(task, code, context, report):
    """assembler tous ces éléments en un prompts

    :param task:
    :param code:
    :param context:
    :param report:
    :return:
    """
    pass