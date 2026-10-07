def ask(label):
    """ Fonction qui est responsable de faire l'équivalent de la fin de ask_qwen"""
    if label:
        return input(label).strip()
    else:
        return None

def main():
    """Fonction responsable de la loop pour faire fonctioner lqwn dans le terminal. Elle appelle les autres fonctions"""
    gen_pipe, model, tokenizer = load_qwen()
    log_path = create_log()
    try:
        while True:
            prompt = ask("Prompt (ou 'quit') : ")
            if prompt in {"quit","exit"}:
                break
            elif prompt == None:


if __name__ == "__main__":
    main()