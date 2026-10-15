
"""
Je veux que le modèle puisse recevoir le prompt d'originie, le code cérer, les erreurs (retours et messages terminal) ainis que le RAG pour pouvoir améliorer lui-même sont prompt.
Une fois l'amélioration du prompt fait, je peux regarder le prompt et confirmer ou valider avant que qwen regénère un nouveau code. Je valide le code et on relance les tests.
Tous les prompts sont sauvegardé dans log comme demandé. Je peux réutiliser des éléments de chat.py
"""
import re

def extract_code(answer):
    """Sortir le code du bloc ```python``` de la réponse actuelle de qwem

    """
    # Code inspiré par Claude
    match =  re.search(r"```(?:python)?\s*\n(.*?)```", answer, re.DOTALL)
    # Code JC
    if not match:
        raise ValueError ("Code extraction failed. Check answer")
    return match.group(1)
def load_function(code,name):
    """transformer str venant de extract en fonction python.
    Avertissement de https://docs.python.org/3/builtins/functions.html#exec, il faut s'assurer que le code est ok
    avant d'appeler exec. C'est correct car on supervise le output de qwen.

    :param code:
    :param name:
    :return:
    """
    dic_code={}
    exec(code,dic_code)
    if name not in dic_code:
        raise ValueError(f"La fonction {name} est absente de la réponse de qwen")
    fn = dic_code[name]
    return fn

def format_test_report(result):
    """rendre run_test lisible pour qwen dans son contexte

    :param result:
    :return:
    """
    n_ok = sum(i=="OK" for i in result.values())
    parts = [f"{n_ok}/{len(result)} tests passed"]
    # run_test renvoie {nom:statut}
    for name, status in result.items():
        parts.append(f"Test: {name}, Status: {status}")
    report = "\n".join(parts)
    return report

def build_refine_prompt(task, code, context, report):
    """assembler tous ces éléments en un prompts

    :param task:
    :param code:
    :param context:
    :param report:
    :return:
    """
    # https://huggingface.co/learn/cookbook/advanced_rag pour les instructions et structure
    text = f"Here is the original task you received.\n<task>{task}</task>\n"
    text = (
        "".join(
            text +
            f"Here is the code previously generated.\n<generated_code>{code}</generated_code>\n"
        )
        if code
        else text
    )
    text = (
        "".join(
            text
            + f"Here are the results of the tests.\n<test_results>{report}</test_results>\n"
        )
        if report
        else text
    )
    text = (
        "".join(
            text
            + f"Rewrite the task prompt so that the code generated from it passes all the tests.\n"
            + f"Keep every requirement of the original task prompt.\n"
            + f"Do not write any code. "
            + f"Return only the new task prompt inside <new_prompt></new_prompt> tags."
        )
        if context
        else text
    )
    text = "".join(text + f"<context>{context}</context>") if context else text

    return text