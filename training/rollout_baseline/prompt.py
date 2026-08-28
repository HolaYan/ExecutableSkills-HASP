"""Direct-answer / single-step CoT prompt template.

The model gets the problem once, thinks step-by-step internally, and emits
ONE final code block. No tools, no skills, no retries. Output is fenced as
```python ... ``` so `CodeAnswerEvaluator.extract` can pull it cleanly.
"""

SYSTEM = (
    "You are an expert Python programmer. Solve the user's coding problem.\n"
    "Think step-by-step about the algorithm, edge cases, and complexity, then "
    "write the final implementation.\n"
    "Output exactly ONE Python code block (```python ... ```) containing the "
    "complete solution. The code must define the requested function/class and "
    "be self-contained (include any imports it needs). Do NOT include example "
    "calls, prints, or test harnesses."
)


def build_user_prompt(question: str, starter_code: str = "") -> str:
    parts = ["Problem:\n", question.strip(), "\n"]
    if starter_code and starter_code.strip():
        parts.append("\nFunction signature / starter code:\n```python\n")
        parts.append(starter_code.rstrip())
        parts.append("\n```\n")
    parts.append(
        "\nThink through the problem step-by-step, then provide the final "
        "implementation in a single ```python ... ``` block."
    )
    return "".join(parts)


def build_messages(row: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_user_prompt(
            row["question"], row.get("starter_code", "")
        )},
    ]
