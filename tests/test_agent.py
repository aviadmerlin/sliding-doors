"""Tests for the AI agent (src/agent.py, src/agent_tools.py).

The chat client is mocked for the whole suite except
test_real_end_to_end_against_deployment, which is auto-skipped unless
.env actually has real Azure OpenAI credentials -- no network calls,
no cost, for the default `pytest` run.
"""

import json
import os

import pytest
from agent_framework import BaseChatClient, ChatResponse, Content, Message
from agent_framework._tools import FunctionInvocationLayer
from dotenv import load_dotenv

import src.agent as agent_module
from src.agent import ToolCall, answer_question, generate_recap, ungrounded_numbers
from src.agent_tools import MatchContext

# Must happen before the skipif below is evaluated (at collection time), so a
# filled-in .env is actually visible to _has_real_credentials().
load_dotenv()


class FakeChatClient(FunctionInvocationLayer, BaseChatClient):
    """A scripted chat client: returns the next ChatResponse off a queue
    each time it's called, and really executes any function_call content
    via Agent Framework's own FunctionInvocationLayer (not mocked) -- so
    these tests exercise the REAL tool functions in src/agent_tools.py,
    only the "model" is fake. Raises if asked for more responses than
    scripted, so an unexpected extra model call fails loudly."""

    def __init__(self, responses: list[ChatResponse]):
        super().__init__()
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    async def _inner_get_response(self, *, messages, stream, options=None, **kwargs):
        self.calls.append(list(messages))
        if stream:
            raise NotImplementedError("streaming not used in these tests")
        if not self._responses:
            raise AssertionError(f"FakeChatClient has no scripted response left (call #{len(self.calls)})")
        return self._responses.pop(0)


def text_response(text: str) -> ChatResponse:
    return ChatResponse(messages=[Message(role="assistant", contents=[Content(type="text", text=text)])])


def function_call_response(name: str, arguments: dict, call_id: str = "call_1") -> ChatResponse:
    return ChatResponse(messages=[Message(role="assistant", contents=[
        Content(type="function_call", call_id=call_id, name=name, arguments=arguments)])])


@pytest.fixture
def match_ctx() -> MatchContext:
    meta = json.loads(open("data/featured_matches.json").read())
    return MatchContext(meta["default_match_id"])


def _patch_chat_client(monkeypatch, client) -> None:
    monkeypatch.setattr(agent_module, "build_chat_client", lambda: client)


# --- Tool routing ---

async def test_tool_routing_calls_win_probability_with_correct_args(match_ctx, monkeypatch) -> None:
    """Scripts the Analyst to call get_win_probability(minute=45), then
    stop; checks the REAL tool actually ran with those arguments and its
    real result (not a mocked one) is what reaches the Narrator."""
    client = FakeChatClient([
        function_call_response("get_win_probability", {"minute": 45}),
        text_response(""),  # Analyst's post-tool note: empty (normal case)
        text_response("At minute 45 the home side has a real chance."),  # Narrator's reply
    ])
    _patch_chat_client(monkeypatch, client)

    answer = await answer_question(match_ctx, "What are the chances at half time?")

    assert len(answer.tool_calls) == 1
    call = answer.tool_calls[0]
    assert call.name == "get_win_probability"
    assert call.arguments == {"minute": 45}
    # The tool really ran: its result (JSON-serialized by Agent Framework)
    # must match calling it directly.
    assert json.loads(call.result) == match_ctx.get_win_probability(45)
    assert answer.text == "At minute 45 the home side has a real chance."


async def test_tool_routing_calls_counterfactual_with_correct_args(match_ctx, monkeypatch) -> None:
    events = match_ctx.list_match_events()
    a_goal = next(e for e in events if e["kind"] == "goal")

    client = FakeChatClient([
        function_call_response("list_match_events", {}, call_id="call_1"),
        function_call_response("run_counterfactual",
                                {"minute": a_goal["minute"], "change": "goal_cancelled", "team": a_goal["team"]},
                                call_id="call_2"),
        text_response(""),
        text_response("Cancelling that goal would have changed things a lot."),
    ])
    _patch_chat_client(monkeypatch, client)

    answer = await answer_question(match_ctx, f"What if the {a_goal['minute']}' goal hadn't happened?")

    names = [tc.name for tc in answer.tool_calls]
    assert names == ["list_match_events", "run_counterfactual"]
    cf_call = answer.tool_calls[1]
    assert cf_call.arguments == {"minute": a_goal["minute"], "change": "goal_cancelled", "team": a_goal["team"]}
    expected = match_ctx.run_counterfactual(minute=a_goal["minute"], change="goal_cancelled", team=a_goal["team"])
    assert json.loads(cf_call.result) == expected


# --- Refusal / clarification (no tool call -- the Analyst's note drives the reply) ---

async def test_refusal_for_unsupported_question(match_ctx, monkeypatch) -> None:
    """"What if they defended better" has no matching tool -- the Analyst
    must not call run_counterfactual with a made-up change type; it should
    write an UNSUPPORTED note instead, which the Narrator turns into an
    honest refusal."""
    client = FakeChatClient([
        text_response("UNSUPPORTED: I can't simulate a general tactical change like better defending. "
                      "I can simulate a specific shot becoming a goal, a goal being cancelled, or a "
                      "red card being undone."),
        text_response("I can't simulate a vague tactical change like that, sorry! But I can tell you "
                      "what would've happened if a specific shot, goal, or red card were different."),
    ])
    _patch_chat_client(monkeypatch, client)

    answer = await answer_question(match_ctx, "What if they had defended better in the second half?")

    assert answer.tool_calls == []
    assert answer.analyst_note.startswith("UNSUPPORTED:")
    assert "can't simulate" in answer.text.lower() or "cannot simulate" in answer.text.lower()


async def test_clarification_for_ambiguous_question(match_ctx, monkeypatch) -> None:
    client = FakeChatClient([
        function_call_response("list_match_events", {}),
        text_response("CLARIFY: Two events could match -- which one did you mean, the goal by "
                      "Player A or the goal by Player B?"),
        text_response("Which goal do you mean -- Player A's or Player B's?"),
    ])
    _patch_chat_client(monkeypatch, client)

    answer = await answer_question(match_ctx, "What if that goal hadn't happened?")

    assert [tc.name for tc in answer.tool_calls] == ["list_match_events"]
    assert answer.analyst_note.startswith("CLARIFY:")
    assert "?" in answer.text


# --- Grounding: every number in the answer must come from a tool result ---

def test_grounding_normalizes_percentage_against_raw_tool_result() -> None:
    """0.283 in a tool result must be recognized as matching "28%" in text."""
    tool_calls = [ToolCall(name="get_win_probability", arguments={"minute": 45},
                           result={"p_home": 0.283, "p_draw": 0.2, "p_away": 0.517})]
    assert ungrounded_numbers("At minute 45 there's a 28% chance of a home win.", tool_calls) == set()


def test_grounding_rejects_invented_number() -> None:
    tool_calls = [ToolCall(name="get_win_probability", arguments={"minute": 45},
                           result={"p_home": 0.283, "p_draw": 0.2, "p_away": 0.517})]
    bad = ungrounded_numbers("There's a 95% chance of a home win.", tool_calls)
    assert bad == {0.95}


def test_grounding_handles_hebrew_text() -> None:
    tool_calls = [ToolCall(name="get_win_probability", arguments={"minute": 45},
                           result={"p_home": 0.283, "p_draw": 0.2, "p_away": 0.517})]
    hebrew_text = "בדקה ה-45 יש סיכוי של 28% לניצחון הקבוצה הביתית, לעומת 20% לתיקו."
    assert ungrounded_numbers(hebrew_text, tool_calls) == set()

    hebrew_text_invented = "בדקה ה-45 יש סיכוי של 99% לניצחון."
    assert ungrounded_numbers(hebrew_text_invented, tool_calls) == {0.99}


async def test_full_answer_is_grounded_in_tool_results(match_ctx, monkeypatch) -> None:
    """End-to-end (mocked model, real tools): the Narrator's reply, scripted
    here to directly reflect the real tool result, must come back fully
    grounded -- exercising the same check answer_question logs a warning
    for internally."""
    probs = match_ctx.get_win_probability(60)
    home_pct = round(probs["p_home"] * 100)

    client = FakeChatClient([
        function_call_response("get_win_probability", {"minute": 60}),
        text_response(""),
        text_response(f"At the hour mark, {match_ctx.match.home_team} have about a {home_pct}% chance to win."),
    ])
    _patch_chat_client(monkeypatch, client)

    answer = await answer_question(match_ctx, "What are the odds at minute 60?")
    assert ungrounded_numbers(answer.text, answer.tool_calls) == set()


# --- Match recap (no question-routing, straight to the Narrator) ---

async def test_recap_is_grounded_in_turning_point_data(match_ctx, monkeypatch) -> None:
    recap_data = match_ctx.get_match_recap_data()
    tp = recap_data["turning_points"][0]
    after_pct = round(tp["after"] * 100)

    client = FakeChatClient([
        text_response(f"{recap_data['home_team']} {recap_data['home_goals']}-{recap_data['away_goals']} "
                      f"{recap_data['away_team']}. The {tp['minute']}' {tp['event_kind']} swung it to "
                      f"{after_pct}%."),
    ])
    _patch_chat_client(monkeypatch, client)

    recap_text = await generate_recap(match_ctx)
    recap_tool_call = ToolCall(name="get_match_recap_data", arguments={}, result=recap_data)
    assert ungrounded_numbers(recap_text, [recap_tool_call]) == set()


# --- One real end-to-end check against the actual deployment ---

def _has_real_credentials() -> bool:
    return bool(os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
                and os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
                and os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "").strip())


@pytest.mark.skipif(not _has_real_credentials(),
                     reason="AZURE_OPENAI_* not set in .env -- real end-to-end test skipped")
async def test_real_end_to_end_against_deployment(match_ctx) -> None:
    """The one test that actually talks to the deployed model. Only runs
    once .env is filled in; asks a simple, unambiguous question and checks
    a real tool was called and the reply is grounded in its result."""
    answer = await answer_question(match_ctx, "What was the win probability at half time?")
    assert len(answer.tool_calls) >= 1
    assert answer.text
    assert ungrounded_numbers(answer.text, answer.tool_calls) == set()
