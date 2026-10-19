
"""
Prompt refinement loop: Qwen receives the original prompt, the generated code, the test results and the RAG
context, and proposes an improved task prompt. I review and approve the new prompt before Qwen regenerates the
code, then the tests are run again. All prompts are saved in the log.

Source 1: https://huggingface.co/learn/cookbook/advanced_rag, Hugging Face cookbook (Advanced RAG)
Source 2: https://docs.python.org/3/library/functions.html#exec, Python documentation (exec)
Source 3: Claude (Anthropic), suggestions in chat, see comments
"""
import re

def extract_text(answer,tag = "code"):
    """Extract the code block or the <new_prompt> block from a Qwen answer.

    :param answer: Qwen answer
    :type answer: str
    :param tag: "code" or "new_prompt"
    :type tag: str
    :return: Extracted text
    :rtype: str
    :raises ValueError: If nothing is found or the tag is unknown
    """
    if tag == "code":
        # Regex inspired by CLAUDE
        match =  re.search(r"```(?:python)?\s*\n(.*?)```", answer, re.DOTALL)
        if not match:
            raise ValueError ("Code extraction failed. Check answer")
    elif tag == "new_prompt":
        match = re.search(r"<new_prompt>(.*?)</new_prompt>", answer, re.DOTALL)
        if not match:
            raise ValueError("New prompt extraction failed. Check new prompt")
    else:
        raise ValueError("Wrong variable type in extract_text(answer,type)")

    return match.group(1)

def load_function(code,name):
    """Turn the extracted code into a callable Python function.

    Warning from source 2: the code must be checked before calling exec. Fine here because
    Qwen's output is reviewed first.

    :param code: Python code
    :type code: str
    :param name: Function name to return
    :type name: str
    :return: The function
    :rtype: callable
    :raises ValueError: If the function is not defined in the code
    """
    dic_code={}
    # cuts our text into function(s). dic_code retain text before def(), the function name
    exec(code,dic_code)
    if name not in dic_code:
        raise ValueError(f"Function {name} is missing from Qwen's answer")
    fn = dic_code[name]
    return fn

def format_test_report(result):
    """Turn the run_tests output into a readable report for Qwen's context.

    :param result: Output of run_tests
    :type result: dict
    :return: Report
    :rtype: str
    """
    n_ok = sum(i=="OK" for i in result.values())
    parts = [f"{n_ok}/{len(result)} tests passed"]
    # run_tests returns {name: status}
    for name, status in result.items():
        parts.append(f"Test: {name}, Status: {status}")
    report = "\n".join(parts)
    return report

def build_refine_prompt(task, code, context, report):
    """Assemble the task, generated code, test report and RAG context into one refine prompt.

    :param task: Original task prompt
    :type task: str
    :param code: Generated code
    :type code: str
    :param context: RAG context
    :type context: str
    :param report: Output of format_test_report
    :type report: str
    :return: Refine prompt
    :rtype: str
    """
    # Structure and instructions adapted from source 1
    parts = [
        f"Here is the original task prompt.\n<task>{task}</task>",
        f"Here is the code generated from this task prompt.\n<generated_code>{code}</generated_code>",
        f"Here is the official API documentation.\n<context>{context}</context>",
        f"Here are the results of the tests run on this code.\n<test_results>{report}</test_results>",
        "Rewrite the task prompt so that the code generated from it passes all the tests. "
        "Keep every requirement of the original task prompt. "
        "Do not write any code. "
        "Return only the new task prompt inside <new_prompt></new_prompt> tags.",
    ]
    return "\n\n".join(parts)

def all_passed(test_results):
    """Check if every test passed.

    :param test_results: Output of run_tests
    :type test_results: dict
    :return: True if every status is "OK"
    :rtype: bool
    """
    return all(status == "OK" for status in test_results.values())
