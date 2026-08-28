"""Mining real rollouts into the material a checker can be written against.

`mine_error_cases.py` (math) and `mine_web_code.py` (web + code) split rollouts
into a wrong set and a correct set — the second is the false-positive control
that decides every downstream gate. `analyze_blindspots.py` reports which error
families the existing library already fires on without changing anything;
`code_error_taxonomy.py` re-judges failing code in the sandbox with the real
test driver.

The math corpus `forge` consumes, `data/llm_anchor/cases.jsonl`, is built by
`anchor/llm_locate.py --stage build`.
"""
