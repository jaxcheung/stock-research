# src/llm/backend.py
#
# One interface for the agents: complete(agent, system, prompt, schema) -> LLMResult.
# Agents receive their data ASSEMBLED INTO THE PROMPT by deterministic code (every number
# comes from the app's own functions) and return JSON validated against a schema — no
# model-side tools, so there is nothing to hallucinate a figure from and nothing to execute.
#
# Backends:
#   claude_code    headless `claude -p` on the user's Claude subscription (no API key).
#                  Called LEAN: own system prompt, --tools "", no session persistence, run
#                  from an empty directory (no CLAUDE.md auto-load), schema in the prompt —
#                  measured 2026-11-07: 572 tokens / $0.0046 list for a trivial call vs
#                  ~18.6k tokens / $0.078 with Claude Code's default prompt and tools.
#   anthropic_api  Anthropic SDK (separately billed). Not wired yet: needs the `anthropic`
#                  package and ANTHROPIC_API_KEY; adds prompt-caching control + Batch API.
# Every call, success or failure, is written to the usage ledger (src/llm/usage.py).

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import jsonschema

from src.config import LLM_BACKEND, LLM_MODEL, LLM_TIMEOUT_SECONDS
from src.llm.usage import CallUsage, record, usage_from_claude_code
from src.utils.logging import get_logger

logger = get_logger("llm")


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    data: Any | None  # parsed structured output when a schema was given
    usage: CallUsage


def _child_env() -> dict[str, str]:
    """Environment for the headless call without the PARENT Claude Code session's variables
    (session id, messaging socket/token, child-session flags): a scheduled agent run must be
    an independent session, never attached to whichever interactive session launched it.
    Authentication is unaffected (the subscription login is stored on disk)."""
    return {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("CLAUDECODE", "CLAUDE_CODE_", "CLAUDE_JOB", "CLAUDE_PID"))
    }


def _strip_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _claude_code(
    agent: str, system: str, prompt: str, schema: dict[str, Any] | None, model: str, run_id: str | None
) -> LLMResult:
    exe = shutil.which("claude")
    if exe is None:
        raise LLMError("`claude` CLI not found — install Claude Code or set LLM_BACKEND=anthropic_api")
    cmd = [
        exe,
        "-p",
        prompt,
        "--output-format",
        "json",
        "--system-prompt",
        system,
        "--tools",
        "",
        "--no-session-persistence",
        "--model",
        model,
    ]
    # The schema goes in the PROMPT and is validated locally: `--json-schema` switches on extra
    # Claude Code machinery (measured 15,001 vs 664 tokens for the same trivial call, 20x cost).
    if schema is not None:
        cmd[2] = (
            prompt
            + "\n\nRespond with ONLY one JSON object (no prose, no code fence) that validates against this "
            + "JSON Schema:\n"
            + json.dumps(schema, sort_keys=True)
        )
    usage = CallUsage(
        agent=agent, backend="claude_code", model=model, prompt_sha256=_sha(system + prompt), run_id=run_id
    )
    start = time.monotonic()
    try:
        # Run from an empty directory: inside the project, Claude Code auto-loads CLAUDE.md and
        # memory files into every call (measured 14,583 vs 664 input tokens for the same call).
        with tempfile.TemporaryDirectory(prefix="llm-call-") as cwd:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=LLM_TIMEOUT_SECONDS,
                check=False,
                cwd=cwd,
                env=_child_env(),
            )
        result = json.loads(proc.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        usage.is_error, usage.error = True, f"{type(exc).__name__}: {str(exc)[:200]}"
        usage.duration_ms = int((time.monotonic() - start) * 1000)
        record(usage)
        raise LLMError(f"claude -p failed for {agent}: {usage.error}") from None
    for k, v in usage_from_claude_code(result).items():
        if v is not None:
            setattr(usage, k, v)
    if result.get("is_error") or result.get("subtype") != "success":
        usage.is_error = True
        usage.error = str(result.get("api_error_status") or result.get("subtype"))[:200]
        record(usage)
        raise LLMError(f"claude -p returned an error for {agent}: {usage.error}")
    text = str(result.get("result") or "")
    data = None
    if schema is not None:
        try:
            data = json.loads(_strip_fence(text))
            jsonschema.validate(data, schema)
        except (json.JSONDecodeError, jsonschema.ValidationError) as exc:
            usage.is_error, usage.error = True, f"invalid structured output: {str(exc)[:160]}"
    record(usage)
    if usage.is_error:
        raise LLMError(f"{agent}: {usage.error}")
    return LLMResult(text=text, data=data, usage=usage)


def complete(
    agent: str,
    system: str,
    prompt: str,
    schema: dict[str, Any] | None = None,
    model: str | None = None,
    run_id: str | None = None,
    backend: str | None = None,
) -> LLMResult:
    """One LLM call, logged to the usage ledger. `agent` names the caller for the budget review."""
    if not agent or not system or not prompt:
        raise ValueError("agent, system and prompt are required")
    which = backend or LLM_BACKEND
    if which == "claude_code":
        return _claude_code(agent, system, prompt, schema, model or LLM_MODEL, run_id)
    if which == "anthropic_api":
        raise LLMError(
            "anthropic_api backend not wired yet — `pip install anthropic`, set ANTHROPIC_API_KEY, and implement "
            "it with prompt caching (stable system prompt first) and the Batch API for monthly runs"
        )
    raise ValueError(f"LLM backend must be 'claude_code' or 'anthropic_api', got {which!r}")
