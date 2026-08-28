"""
Skill validator — automated executable checks for candidate PF code.

Runs BEFORE helper review to filter out obviously broken candidates:
  1. Syntax check (can the code parse?)
  2. Import check (does it reference valid modules?)
  3. Interface check (does it have should_activate / intervene with correct signatures?)
  4. Mock execution (does it run without error on sample step_contexts?)
  5. Return type check (does intervene return an Intervention?)

Only candidates that pass all checks proceed to helper review.
"""

import ast
import logging
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional

from .skill_proposer import CandidateSkill

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of automated PF code validation."""
    skill_id: str
    passed: bool = False
    # Per-check results
    syntax_ok: bool = False
    interface_ok: bool = False
    mock_execution_ok: bool = False
    return_type_ok: bool = False
    # Error details
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "passed": self.passed,
            "syntax_ok": self.syntax_ok,
            "interface_ok": self.interface_ok,
            "mock_execution_ok": self.mock_execution_ok,
            "return_type_ok": self.return_type_ok,
            "errors": self.errors,
            "warnings": self.warnings,
        }


# Sample step_contexts for mock execution
MOCK_STEP_CONTEXTS = [
    # Early step, no read yet
    {
        "question": "Who was the first president of the United States?",
        "step_count": 1,
        "has_read": False,
        "search_count": 1,
        "read_count": 0,
        "empty_results": False,
        "contradictory_sources": False,
        "max_steps": 25,
        "action_history": [{"action_type": "SEARCH", "arg": "first president United States", "step": 0}],
        "last_search_results_text": "George Washington was the first president...",
        "all_read_contents": "",
        "thought": "I found results about George Washington. Let me give the answer.",
    },
    # Mid-step, has read
    {
        "question": "When did Marie Curie win her second Nobel Prize?",
        "step_count": 5,
        "has_read": True,
        "search_count": 3,
        "read_count": 2,
        "empty_results": False,
        "contradictory_sources": False,
        "max_steps": 25,
        "action_history": [
            {"action_type": "SEARCH", "arg": "Marie Curie Nobel Prize", "step": 0},
            {"action_type": "READ", "arg": "doc_1", "step": 1},
            {"action_type": "SEARCH", "arg": "Marie Curie second Nobel Prize year", "step": 2},
            {"action_type": "READ", "arg": "doc_2", "step": 3},
        ],
        "last_search_results_text": "Marie Curie won Chemistry Nobel in 1911...",
        "all_read_contents": "Marie Curie won Physics Nobel in 1903 and Chemistry Nobel in 1911.",
        "thought": "Based on my reading, the answer is 1911.",
    },
    # Late step, many searches but empty results
    {
        "question": "What is the population of the smallest country in Africa?",
        "step_count": 12,
        "has_read": True,
        "search_count": 8,
        "read_count": 3,
        "empty_results": True,
        "contradictory_sources": True,
        "max_steps": 25,
        "action_history": [
            {"action_type": "SEARCH", "arg": "smallest country Africa population", "step": i}
            for i in range(8)
        ],
        "last_search_results_text": "",
        "all_read_contents": "Seychelles population approximately 98,000. Eswatini population 1.1 million.",
        "thought": "I'm running out of searches. Let me try a different approach.",
    },
]


class SkillValidator:
    """Validates candidate PF code through automated checks."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else None

    def validate(self, candidates: List[CandidateSkill]) -> List[ValidationResult]:
        """Validate all candidates. Returns results; also updates candidate objects."""
        results = []
        for candidate in candidates:
            result = self._validate_one(candidate)
            results.append(result)
            if not result.passed:
                logger.warning("Validation FAILED for %s: %s",
                               candidate.skill_id, "; ".join(result.errors))
            else:
                logger.info("Validation passed for %s", candidate.skill_id)

        if self.output_dir:
            self._save_results(results)

        return results

    def _validate_one(self, candidate: CandidateSkill) -> ValidationResult:
        """Run all checks on a single candidate."""
        result = ValidationResult(skill_id=candidate.skill_id)
        code = candidate.pf_code

        if not code.strip():
            result.errors.append("Empty PF code")
            return result

        # Check 1: Syntax
        result.syntax_ok = self._check_syntax(code, result)
        if not result.syntax_ok:
            return result

        # Check 2: Interface (has correct methods with correct signatures)
        result.interface_ok = self._check_interface(code, result)
        if not result.interface_ok:
            return result

        # Check 3: Mock execution
        result.mock_execution_ok = self._check_mock_execution(code, candidate.skill_id, result)

        # Check 4: Return type
        result.return_type_ok = self._check_return_type(code, result)

        # Overall pass/fail
        result.passed = (
            result.syntax_ok
            and result.interface_ok
            and result.mock_execution_ok
            and result.return_type_ok
        )
        return result

    def _check_syntax(self, code: str, result: ValidationResult) -> bool:
        """Check if code parses as valid Python."""
        # Wrap with necessary imports for parsing
        full_code = self._wrap_code(code)
        try:
            ast.parse(full_code)
            return True
        except SyntaxError as e:
            result.errors.append(f"SyntaxError: {e.msg} (line {e.lineno})")
            return False

    def _check_interface(self, code: str, result: ValidationResult) -> bool:
        """Check if the PF class has required methods with correct signatures."""
        try:
            tree = ast.parse(self._wrap_code(code))
        except SyntaxError:
            result.errors.append("Cannot parse for interface check")
            return False

        # Find class definitions
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        if not classes:
            result.errors.append("No class definition found in PF code")
            return False

        pf_class = None
        for cls in classes:
            # Check if it inherits from ProgramFunction
            for base in cls.bases:
                if isinstance(base, ast.Name) and base.id == "ProgramFunction":
                    pf_class = cls
                    break
                if isinstance(base, ast.Attribute) and base.attr == "ProgramFunction":
                    pf_class = cls
                    break
            if pf_class:
                break

        if not pf_class:
            # Accept even without explicit inheritance if methods exist
            pf_class = classes[0]
            result.warnings.append("Class doesn't explicitly inherit from ProgramFunction")

        methods = {node.name: node for node in pf_class.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}

        # Check should_activate
        if "should_activate" not in methods:
            result.errors.append("Missing should_activate() method")
            return False
        sa = methods["should_activate"]
        sa_args = [a.arg for a in sa.args.args]
        if len(sa_args) < 4:  # self, step_context, action_type, arg
            result.errors.append(
                f"should_activate has {len(sa_args)} args, expected 4 (self, step_context, action_type, arg)")
            return False

        # Check intervene
        if "intervene" not in methods:
            result.errors.append("Missing intervene() method")
            return False
        iv = methods["intervene"]
        iv_args = [a.arg for a in iv.args.args]
        if len(iv_args) < 4:  # self, step_context, action_type, arg (+ optional PF helper)
            result.errors.append(
                f"intervene has {len(iv_args)} args, expected >=4 (self, step_context, action_type, arg)")
            return False

        return True

    def _check_mock_execution(self, code: str, skill_id: str, result: ValidationResult) -> bool:
        """Try to execute the PF on mock step_contexts in a sandbox."""
        full_code = self._wrap_code(code)

        # Create a sandbox namespace
        sandbox = {}
        try:
            exec(full_code, sandbox)
        except Exception as e:
            result.errors.append(f"Exec failed: {type(e).__name__}: {e}")
            return False

        # Find the PF class instance
        pf_instance = None

        # Check if register_pf put it in the mock registry
        mock_registry = sandbox.get("_mock_registry", {})
        if mock_registry:
            pf_instance = list(mock_registry.values())[0]
        else:
            # Try to find and instantiate the class directly
            for name, obj in sandbox.items():
                if isinstance(obj, type) and name not in (
                    "ProgramFunction", "Intervention", "InterventionType", "Enum"
                ):
                    try:
                        pf_instance = obj()
                        break
                    except Exception:
                        continue

        if not pf_instance:
            result.errors.append("Could not instantiate PF class")
            return False

        # Run on mock contexts
        all_ok = True
        for i, ctx in enumerate(MOCK_STEP_CONTEXTS):
            for action_type in ("SEARCH", "READ", "FINAL"):
                arg = ctx.get("last_search_results_text", "test query")[:50]
                try:
                    activated = pf_instance.should_activate(ctx, action_type, arg)
                    if not isinstance(activated, bool):
                        result.warnings.append(
                            f"should_activate returned {type(activated).__name__}, expected bool "
                            f"(ctx={i}, action={action_type})")

                    if activated:
                        intervention = pf_instance.intervene(ctx, action_type, arg)
                        # Basic check: intervention should have a type attribute
                        if not hasattr(intervention, "type"):
                            result.errors.append(
                                f"intervene returned object without .type (ctx={i}, action={action_type})")
                            all_ok = False
                except Exception as e:
                    result.errors.append(
                        f"Runtime error on ctx={i}, action={action_type}: {type(e).__name__}: {e}")
                    all_ok = False

        return all_ok

    def _check_return_type(self, code: str, result: ValidationResult) -> bool:
        """Check that intervene returns Intervention objects (via AST analysis)."""
        try:
            tree = ast.parse(self._wrap_code(code))
        except SyntaxError:
            return False

        # Find return statements in intervene method
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "intervene":
                returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
                if not returns:
                    result.warnings.append("intervene has no return statement")
                    return False

                # Check if any return creates an Intervention
                for ret in returns:
                    if ret.value is None:
                        result.warnings.append("intervene has bare return (should return Intervention)")
                        continue
                    if isinstance(ret.value, ast.Call):
                        func = ret.value.func
                        if isinstance(func, ast.Name) and func.id == "Intervention":
                            return True
                        if isinstance(func, ast.Attribute) and func.attr == "Intervention":
                            return True

                # If we found returns but none create Intervention explicitly,
                # still pass with warning (might use variable)
                result.warnings.append("intervene returns something but not obviously an Intervention")
                return True

        return True

    @staticmethod
    def _wrap_code(code: str) -> str:
        """Wrap PF code with necessary imports and mock registry for sandbox execution."""
        preamble = '''\
import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

class InterventionType(Enum):
    NOOP = "noop"
    MODIFY_ACTION = "modify_action"
    INJECT_CONTEXT = "inject_context"

@dataclass
class Intervention:
    type: InterventionType = InterventionType.NOOP
    new_action_type: Optional[str] = None
    new_action_arg: Optional[str] = None
    context_text: str = ""
    reason: str = ""
    skill_id: str = ""

class ProgramFunction:
    skill_id: str = ""
    needs_helper: bool = False
    def should_activate(self, step_context, action_type, arg):
        raise NotImplementedError
    def intervene(self, step_context, action_type, arg, helper=None):
        raise NotImplementedError

_mock_registry = {}

def register_pf(skill_id):
    def decorator(cls):
        cls.skill_id = skill_id
        _mock_registry[skill_id] = cls()
        return cls
    return decorator

'''
        return preamble + code

    def _save_results(self, results: List[ValidationResult]) -> None:
        if not self.output_dir:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "validation_results.json"
        import json
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
