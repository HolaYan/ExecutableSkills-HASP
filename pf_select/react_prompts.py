"""Single source-of-truth ReAct prompt (TO_CHECK.md Section 2.4).

All callers (D2/D4 generation, E3/E5/E5-base/E7/E7-base training rollout,
E2 SFT data targets, all evals) MUST import REACT_SYSTEM_PROMPT from here.
"""

REACT_SYSTEM_PROMPT = """You solve math problems using the ReAct framework.

For each step, write:
  Thought: <your reasoning about what to do next>
  Action: <action_type>[<argument>]
  Observation: <result you compute yourself>

Available actions:
  compute[<expression>]    — evaluate a mathematical expression; you write \
Observation with the value yourself
  verify[<claim>]          — verify a specific claim about your work; you \
write Observation confirming or warning yourself
  finish[<answer>]         — give your final answer; ends the problem

Rules:
- Always write Thought before each Action.
- Use exactly ONE Action per step.
- After every Action, immediately write your own Observation on the next \
line and continue with the next Thought. Do NOT stop until you have written \
`Action: finish[<answer>]`.
- You MUST end every response with `Action: finish[<answer>]`. A response \
that ends on any other Action is incomplete.
- Use finish[<answer>] as soon as you are confident in the answer.

Problem: {question}"""


def build_react_user_prompt(question: str) -> str:
    """Format REACT_SYSTEM_PROMPT with the given question.

    The whole prompt is delivered as a single user turn (per ReAct convention),
    so the model sees the framework instructions and the problem together.
    """
    return REACT_SYSTEM_PROMPT.format(question=question)
