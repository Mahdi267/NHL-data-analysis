"""
Interactive terminal chat with the local Qwen model. Each call is logged in a JSONL file. Was not used as intented but a
fun experiment and a great tutorial.
"""

from ift6758.llm.qwen_utils import build_messages, load_qwen, generate, free_qwen
from ift6758.llm.qwen_log import create_log, log_call


def ask_user(label):
    """Ask the user for text input.

    :param label: Text shown before the input
    :type label: str
    :return: User answer, None if empty
    :rtype: str or None
    """
    answer = input(label).strip()
    return answer if answer else None

def confirm(label):
    """Ask a yes/no question.

    :param label: Question shown to the user
    :type label: str
    :return: True if the answer is o/oui/y/yes
    :rtype: bool
    """
    answer = ask_user(f"{label} (y/n) : ")
    return answer is not None and answer.lower() in {"o", "oui", "y", "yes"}


def main():
    """Load Qwen once, then loop on prompt -> answer -> log until quit/exit, Ctrl+C or Ctrl+D.

    :return: None
    """
    model, tokenizer, device = load_qwen()
    log_path = create_log()
    try:
        while True:
            prompt = ask_user("Prompt (or 'quit') : ")
            # Avoid running the model without a prompt by mistake
            if prompt in {"quit", "exit"}:
                break
            if prompt is None:
                continue
            system = ask_user("Instructions : ")
            context = ask_user("Context : ")
            messages = build_messages(prompt, context, system)
            result = generate(model, tokenizer, messages)
            print(
                f"Answer: {result['answer']} \n"
                f"Input tokens : {result['n_token_in']} \n"
                f"Generated tokens : {result['n_token_out']} \n"
                f"Duration : {result['duration_s']} \n"
            )
            log_call(log_path, messages, result)

    except (KeyboardInterrupt, EOFError):
        print("Stop requested")
    # Stop Qwen and free the memory
    finally:
        model = tokenizer = (
            None  # Drop the references so the Qwen objects can be freed
        )
        # del model, tokenizer
        free_qwen(device)
        print(f"Conversation recorded. Log path: {log_path}")


if __name__ == "__main__":
    main()
