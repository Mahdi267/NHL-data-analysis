
"""
Je veux que le modèle puisse recevoir le prompt d'originie, le code cérer, les erreurs (retours et messages terminal) ainis que le RAG pour pouvoir améliorer lui-même sont prompt.
Une fois l'amélioration du prompt fait, je peux regarder le prompt et confirmer ou valider avant que qwen regénère un nouveau code. Je valide le code et on relance les tests.
Tous les prompts sont sauvegardé dans log comme demandé. Je peux réutiliser des éléments de chat.py
"""
import re

def extract_text(answer,tag = "code"):
    """Sortir le code du bloc ```python``` de la réponse actuelle de qwem

    """
    if tag == "code":
        # Code inspiré par Claude
        match =  re.search(r"```(?:python)?\s*\n(.*?)```", answer, re.DOTALL)
        # Code JC
        if not match:
            raise ValueError ("Code extraction failed. Check answer")
    elif tag == "new_prompt":
        match = re.search(r"<new_prompt>(.*?)</new_prompt>", answer, re.DOTALL)
        # Code JC
        if not match:
            raise ValueError("New prompt extraction failed. Check new prompt")
    else:
        raise ValueError("Wrong variable type in extract_text(answer,type)")
        
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
    """Assemble la tâche, le code généré, le rapport de tests et la doc RAG
    en un message pour que Qwen propose un nouveau prompt de tâche.
    :param task:
    :param code:
    :param context:
    :param report:
    :return:
    """
    # https://huggingface.co/learn/cookbook/advanced_rag pour les instructions et structure
    parts = [
        f"Here is the original task prompt.\n<task>{task}</task>",
        f"Here is the code generated from this task prompt.\n<generated_code>{code}</generated_code>",
        f"Here are the results of the tests run on this code.\n<test_results>{report}</test_results>",
        f"Here is the official API documentation.\n<context>{context}</context>",
        "Rewrite the task prompt so that the code generated from it passes all the tests. "
        "Keep every requirement of the original task prompt. "
        "Do not write any code. "
        "Return only the new task prompt inside <new_prompt></new_prompt> tags."
    ]
    return "\n\n".join(parts)