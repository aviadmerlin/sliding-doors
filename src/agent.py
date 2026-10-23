"""The AI agent: an Analyst (tool use, facts only) and a Narrator
(writing style, tone, language), built on Microsoft Agent Framework.

Two roles, cleanly separated:
  - Analyst: sees the fan's question and the four tools in src/agent_tools.py.
    It either calls the tool(s) it needs, or -- when the question is
    ambiguous or asks for something the engine can't simulate -- writes one
    short plain note (never the fan-facing answer) saying so.
  - Narrator: always writes the final reply, in the requested tone (fan /
    analyst) and language (English / Hebrew), from whatever the Analyst
    produced. It is explicitly instructed to use ONLY numbers present in
    the tool results it's given -- never to invent or recompute one.

Provider access is behind build_chat_client() -- the one place that
constructs a concrete client -- so swapping providers later doesn't touch
the Analyst/Narrator logic or the tools themselves.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Literal

from agent_framework import Agent

from src.agent_tools import MatchContext

logger = logging.getLogger(__name__)

Mode = Literal["fan", "analyst"]
Language = Literal["english", "hebrew"]

# Keep cost/latency low: a tool-routing + one-paragraph-of-prose task has no
# need for anything more than the cheapest reasoning tier and a short reply.
# MAX_OUTPUT_TOKENS has to cover gpt-5-mini's hidden reasoning tokens too, not
# just the visible reply -- 300 proved too tight in practice (reasoning alone
# consumed the whole budget, leaving an empty or mid-sentence-truncated
# answer); 1200 leaves headroom for "low" reasoning + a few sentences.
REASONING_EFFORT: Literal["low", "medium", "high"] = "low"
MAX_OUTPUT_TOKENS = 1200

# Plain-language descriptions ONLY -- these (not the internal change-type
# identifiers used in code) are what the Analyst is told to use when
# explaining what run_counterfactual can and can't do, so a refusal never
# leaks an internal name like "shot_becomes_goal" to the fan.
_SUPPORTED_CHANGES_PLAIN = ("turning a specific missed shot or penalty into a goal, "
                           "cancelling a goal that was actually scored, "
                           "or undoing a red card")

ANALYST_INSTRUCTIONS = f"""You are a football match analyst with four tools: list_match_events,
get_win_probability, run_counterfactual, get_club_profile. A fan will ask a question about ONE
specific match that is already fixed for this conversation -- you never need or receive a
match_id.

Rules:
- To act on an event (a goal, a missed chance, a red card) you must know its exact minute and
  team. Call list_match_events first if you don't already know them from earlier in this
  conversation -- never guess a minute or player name.
- ANY question about what WOULD have happened differently to a specific real event (a goal
  scored/cancelled, a shot scoring, a red card undone) is a hypothetical and REQUIRES calling
  run_counterfactual to get real forecast numbers -- this applies no matter what language the
  question is asked in. list_match_events only tells you what REALLY happened; it never tells
  you what would have happened under a hypothetical, and must never be used as a substitute for
  actually calling run_counterfactual. Never describe a hypothetical outcome, a change in score,
  or a change in win chance unless run_counterfactual actually returned those numbers.
- The ONLY changes that can be simulated are: {_SUPPORTED_CHANGES_PLAIN}. If the question asks
  for anything else (a tactical change, a substitution, "what if they defended better", a
  different formation, a different manager, etc.), do NOT call run_counterfactual. Instead:
    1. Call list_match_events first if you haven't already (you need REAL events to suggest).
    2. Write exactly one note starting with "UNSUPPORTED:" that (a) explains in those same plain
       words what CAN be simulated (never mention tool names, function names, or internal
       identifiers like "shot_becomes_goal" -- the fan should never see code-shaped language),
       and (b) suggests exactly 2 concrete example questions built from REAL events in this
       match (real minute, team, player from list_match_events), e.g. "you could ask what if
       [player]'s shot in the 63rd minute had scored" -- not generic placeholders.
- If more than one real event could match the question (e.g. two notable events in the same
  minute, or the question doesn't say which team and both teams had a qualifying event), do NOT
  guess and do NOT call a tool with a guessed team. Instead, write exactly one short plain
  sentence starting with "CLARIFY:" asking which specific event the fan means, naming the real
  candidates (minute, team, player) from list_match_events.
- Otherwise, call whatever tool(s) you need to answer the question with real numbers.
- You are never the one who writes the fan-facing answer -- a separate step does that from your
  tool results. Do not write a conversational reply. Once you've called the tools you need,
  your final message should be empty, UNLESS it's one of the CLARIFY/UNSUPPORTED notes above.
"""


_HEBREW_STYLE = """Respond in Hebrew (עברית), including any numbers. Write like a native Israeli
sports commentator talking naturally -- not a word-for-word translation from English. Use the
phrasing a Hebrew football broadcast would actually use (e.g. "סיכויי ניצחון", "קבוצת הבית" /
"קבוצת החוץ", "דקה ה-X", percentages written as "X אחוז" or "X%"), natural Hebrew sentence order
and idiom, not English syntax carried over word for word."""


def _narrator_instructions(mode: Mode, language: Language) -> str:
    tone = ("Write like a fan-facing broadcast pundit: short, energetic, plain language, no jargon."
            if mode == "fan" else
            "Write like a match-analyst briefing: precise and a little more technical, but still "
            "plain English sentences, no stats jargon left unexplained.")
    lang = "Respond in English." if language == "english" else _HEBREW_STYLE
    return f"""You turn match-analysis facts into a short answer for a fan. {tone} {lang}

CRITICAL rules:
- Use ONLY numbers that appear in the "Tool results" given to you below. Never invent, estimate,
  or recompute a number that isn't already there. If no tool results are given (you're relaying
  a clarifying question or an honest "can't simulate that" note instead), do not invent facts to
  answer anyway -- just deliver that note in the requested tone and language.
- If the fan asked a "what if" question about a specific event but the Tool results do NOT
  include a run_counterfactual-style result (forecast_before_change / forecast_after_change) --
  only a plain list of real events, say so honestly (e.g. "I have the real events but didn't get
  a forecast for that scenario -- try asking again") instead of describing what the hypothetical
  change would have done. A list of real events is never a substitute for a real simulated
  forecast -- do not narrate a hypothetical score or hypothetical win chance that isn't actually
  in the tool results.
- Round every probability to the nearest whole percent when you write it (0.1142 -> "11%", not
  "11.42%" or "0.1142") -- these are simulation estimates, not precise to a fraction of a percent.
- Never mention tool names, function names, or internal identifiers (e.g. "run_counterfactual",
  "get_win_probability", "shot_becomes_goal") -- describe everything in plain fan language.
- If the tool results include a counterfactual (headline_team / headline_before / headline_after),
  LEAD the reply with that fair before-vs-after comparison -- e.g. "Without this goal, Oakhaven's
  win chance would have dropped from 72% to 46%." -- then add ONE short line of context. If the
  real final score / real outcome are also given, you may mention what ACTUALLY happened, but
  state it clearly as the real result (e.g. "In reality, they won 3-2") -- never blend it into
  the before/after forecast comparison as if it were part of the same number.

Keep the reply to 2-4 sentences."""


def build_chat_client():
    """The one place that constructs a concrete chat client -- swap the
    provider here if ever needed, without touching anything else.

    Uses Microsoft Agent Framework's OpenAIChatClient (Responses API)
    configured for Azure routing, since it has first-class, documented
    support for the `reasoning` effort option gpt-5-mini needs for cost
    control (Agent Framework's OTHER Azure-capable client,
    OpenAIChatCompletionClient, uses the older Chat Completions API and
    doesn't expose this option in its typed options)."""
    from agent_framework_openai import OpenAIChatClient

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "").strip()
    if not (endpoint and api_key and deployment):
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_CHAT_DEPLOYMENT_NAME "
            "must all be set (in .env) to use the agent.")
    return OpenAIChatClient(model=deployment, api_key=api_key, azure_endpoint=endpoint)


_tracing_configured = False


def maybe_enable_tracing() -> bool:
    """Enables tracing of agent runs + tool calls to Application Insights
    (visible in the Foundry portal) when APPLICATIONINSIGHTS_CONNECTION_STRING
    is set -- optional, so tests never need it. Safe to call repeatedly;
    only configures the OTel pipeline once per process."""
    global _tracing_configured
    if _tracing_configured:
        return True
    connection_string = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING", "").strip()
    if not connection_string:
        return False
    from agent_framework.observability import enable_instrumentation
    from azure.monitor.opentelemetry import configure_azure_monitor

    configure_azure_monitor(connection_string=connection_string)
    enable_instrumentation()
    _tracing_configured = True
    return True


@dataclass
class ToolCall:
    name: str
    arguments: dict | str
    result: object = None


@dataclass
class AgentAnswer:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    analyst_note: str = ""


def _extract_tool_calls(response) -> list[ToolCall]:
    """Pulls (name, arguments, result) for every tool call out of an
    AgentResponse's message history, in call order -- used both to brief
    the Narrator and, in tests, to check tool routing and that every
    number in the final reply traces back to one of these results."""
    calls_by_id: dict[str, ToolCall] = {}
    order: list[str] = []
    for message in response.messages:
        for content in message.contents:
            if content.type == "function_call":
                calls_by_id[content.call_id] = ToolCall(name=content.name, arguments=content.arguments)
                order.append(content.call_id)
            elif content.type == "function_result" and content.call_id in calls_by_id:
                calls_by_id[content.call_id].result = content.result
    return [calls_by_id[cid] for cid in order]


# --- Grounding: every number in the final text must trace back to a tool result ---

_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(%)?")


def _numbers_in_text(text: str) -> set[float]:
    """Every number mentioned in text, normalized to the same 0-1 scale a
    tool-result probability would be on: "28%" and "0.28" both become 0.28.
    Works on any script (e.g. Hebrew prose) since digits are the same
    Unicode characters regardless of surrounding text direction/language --
    only the regex's match, not any assumption about word order, matters."""
    numbers = set()
    for value_str, is_percent in _NUMBER_RE.findall(text):
        value = float(value_str)
        numbers.add(round(value / 100.0, 2) if is_percent else round(value, 2))
    return numbers


def _numbers_in_value(value: object, out: set[float]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        out.add(round(float(value), 2))
    elif isinstance(value, str):
        # Agent Framework JSON-serializes tool return values before storing
        # them as a function_result's .result -- every real tool call arrives
        # here as a JSON string, not the original dict, so this is the normal
        # case, not a fallback.
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return
        _numbers_in_value(parsed, out)
    elif isinstance(value, dict):
        for v in value.values():
            _numbers_in_value(v, out)
    elif isinstance(value, (list, tuple, set)):
        for v in value:
            _numbers_in_value(v, out)


def _numbers_in_tool_results(tool_calls: list[ToolCall]) -> set[float]:
    """Every numeric leaf value appearing in any tool call's arguments or
    result, normalized the same way as _numbers_in_text so the two sets
    are directly comparable."""
    numbers: set[float] = set()
    for tc in tool_calls:
        _numbers_in_value(tc.arguments, numbers)
        _numbers_in_value(tc.result, numbers)
    return numbers


def ungrounded_numbers(text: str, tool_calls: list[ToolCall], tolerance: float = 0.015) -> set[float]:
    """Numbers mentioned in `text` that don't match (within `tolerance`,
    to absorb rounding to the nearest integer percent) any number actually
    present in the tool call arguments/results -- empty means every number
    in the text is grounded. tolerance=0.015 covers rounding a value to the
    nearest whole percent (worst case 0.5 points = 0.005) with margin."""
    text_numbers = _numbers_in_text(text)
    result_numbers = _numbers_in_tool_results(tool_calls)
    return {n for n in text_numbers
            if not any(abs(n - r) <= tolerance for r in result_numbers)}


def build_analyst(chat_client, ctx: MatchContext) -> Agent:
    return Agent(
        client=chat_client,
        instructions=ANALYST_INSTRUCTIONS,
        tools=ctx.tools(),
        default_options={"reasoning": {"effort": REASONING_EFFORT}, "max_tokens": MAX_OUTPUT_TOKENS},
    )


def build_narrator(chat_client, mode: Mode, language: Language) -> Agent:
    return Agent(
        client=chat_client,
        instructions=_narrator_instructions(mode, language),
        default_options={"reasoning": {"effort": REASONING_EFFORT}, "max_tokens": MAX_OUTPUT_TOKENS},
    )


def _format_tool_results(tool_calls: list[ToolCall]) -> str:
    if not tool_calls:
        return "(no tool was called)"
    return "\n".join(f"- {tc.name}({tc.arguments}) -> {tc.result}" for tc in tool_calls)


async def answer_question(ctx: MatchContext, question: str, mode: Mode = "fan",
                           language: Language = "english") -> AgentAnswer:
    """Routes a fan's free-text question to the right tool call(s) (or a
    clarifying/refusal note), then writes the final styled answer from the
    real results. See module docstring for the two-role design."""
    maybe_enable_tracing()
    chat_client = build_chat_client()

    analyst = build_analyst(chat_client, ctx)
    analyst_response = await analyst.run(question)
    tool_calls = _extract_tool_calls(analyst_response)
    analyst_note = (analyst_response.text or "").strip()

    narrator = build_narrator(chat_client, mode, language)
    narrator_prompt = (
        f"Fan's question: {question}\n\n"
        f"Tool results:\n{_format_tool_results(tool_calls)}\n\n"
        f"Analyst's note (if any): {analyst_note or '(none)'}\n\n"
        "Write the final answer now."
    )
    narrator_response = await narrator.run(narrator_prompt)
    final_text = (narrator_response.text or "").strip()

    bad_numbers = ungrounded_numbers(final_text, tool_calls)
    if bad_numbers and tool_calls:  # only meaningful when there were real results to ground against
        logger.warning("Narrator answer contains numbers not found in any tool result: %s "
                        "(question=%r, text=%r)", bad_numbers, question, final_text)

    return AgentAnswer(text=final_text, tool_calls=tool_calls, analyst_note=analyst_note)


async def generate_recap(ctx: MatchContext, mode: Mode = "fan", language: Language = "english") -> str:
    """A short narrative recap built from the match's turning points and
    final result -- no question to route, so this goes straight to the
    Narrator with the recap data as its only grounding."""
    maybe_enable_tracing()
    chat_client = build_chat_client()
    recap_data = ctx.get_match_recap_data()

    narrator = build_narrator(chat_client, mode, language)
    prompt = ("Write a short match recap (3-5 sentences) using ONLY this data -- do not invent "
              f"any other numbers or events:\n{recap_data}")
    response = await narrator.run(prompt)
    return (response.text or "").strip()
