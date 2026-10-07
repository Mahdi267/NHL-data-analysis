from ift6758.llm.qwen_utils import build_messages,load_qwen,generate, free_qwen
from ift6758.llm.qwen_log import create_log, log_call



def ask_user(label):
    """Ask user to enter text and return user answer

    :param label: Query to user (text in terminal before user input)
    :type label: str
    :return: user answer (None if empty)
    """
    answer = input(label).strip()
    return answer if answer else None


def main():
    """Charge Qwen Une seule fois et créer fichier de log
    Loops on prompt with optionnal system and context, then ask qwen and display its answer, then logs
    Ends on quit and exit prompt or Ctrl+C or Ctrl+D and kills qwen

    :param
    :return None
    """
    gen_pipe, model, tokenizer = load_qwen()
    log_path = create_log()
    try:
        while True:
            prompt = ask_user("Prompt (ou 'quit') : ")
            # Éviter de lancer le modèle sans prompt par erreur
            if prompt in {"quit","exit"}:
                break
            elif prompt == None:
                continue
            system = ""
            context = ""
            messages = build_messages(gen_pipe,model,tokenizer,prompt)
            result = generate(gen_pipe,model,tokenizer,prompt,messages)
            print(result["text"])
            log_call(log_path,result["text"],result["user"])

    except (KeyboardInterrupt,EOFError):
        gen_pipe = model = none
        free_qwen(gen_pipe,model,tokenizer)
        print(f"Conversation recorded. Log path: {log_path}")





if __name__ == "__main__":
    main()