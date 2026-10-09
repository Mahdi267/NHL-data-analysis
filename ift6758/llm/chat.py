from ift6758.llm.qwen_utils import build_messages, load_qwen, generate, free_qwen
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

    :return: None
    """
    model, tokenizer, device = load_qwen()
    log_path = create_log()
    try:
        while True:
            prompt = ask_user("Prompt (ou 'quit') : ")
            # Éviter de lancer le modèle sans prompt par erreur
            if prompt in {"quit", "exit"}:
                break
            if prompt is None:
                continue
            system = ask_user("Instructions : ")
            context = ask_user("Contexte : ")
            messages = build_messages(prompt, context, system)
            result = generate(model, tokenizer, messages)
            print(
                f"Réponse: {result['answer']} \n"
                f"Token entrant : {result['n_token_in']} \n"
                f"Token généré : {result['n_token_out']} \n"
                f"Durée : {result['duration_s']} \n"
            )
            log_call(log_path, messages, result)

    except (KeyboardInterrupt, EOFError):
        print("Arrêt demandé")
    # Ici, on arrête le qwen et on libère la mémoire
    finally:
        model = tokenizer = (
            None  # Ca c'est pour pouvoir supprimer l'objet qwen. On le vide
        )
        # del model, tokenizer
        free_qwen(device)
        print(f"Conversation recorded. Log path: {log_path}")


if __name__ == "__main__":
    main()
