# Sliding Doors — hackathon submission

Microsoft x Premier League "Inside the Game" hackathon. See [README.md](README.md)
for full technical documentation and [SPEC.md](SPEC.md) for the original MVP
checklist.

- **Live demo:** https://sliding-doors.happytree-e6502234.switzerlandnorth.azurecontainerapps.io/
- **Demo video:** _TODO — add the hosted video URL here before submitting_
- **Repo:** https://github.com/aviadmerlin/sliding-doors (public)

## Elevator pitch (~160 words)

Sliding Doors turns football's oldest argument — "if that goal hadn't
gone in, we'd have won" — into a real, data-backed answer. It's a
counterfactual match-intelligence app: a synthetic 20-club league with
hidden traits generates 5 seasons of realistic matches, a weighted
Poisson regression recovers each club's attack, defense, and in-game
tendencies from the data, and a vectorized Monte Carlo engine computes
live win-probability curves and ranks each match's turning points by how
much they really swung the outcome. Fans can replay any moment — cancel
a goal, convert a near-miss, undo a red card — and see a fair
before/after forecast. A two-agent AI layer, built on **Microsoft Agent
Framework** and running **gpt-5-mini** on **Microsoft Foundry**, lets fans
just ask in plain English or Hebrew: an Analyst agent calls the right
simulation tool, a Narrator writes the answer in a fan or analyst tone,
grounding every number in a real result and refusing honestly when a
question can't be simulated.

## Project description

**The problem.** Every football broadcast and fan argument eventually
turns into a counterfactual — "if that penalty had gone in," "if the red
card hadn't happened" — and today that's always just opinion. There's no
tool that turns a specific match moment into an honest, quantified answer
grounded in how the rest of the match plausibly would have gone, and no
way for a fan to just ask that question in their own words and get a
trustworthy reply instead of a hallucinated one.

**What we built.** Sliding Doors is a full pipeline from synthetic data
to a conversational answer:

1. **A synthetic league and match engine** (`src/league.py`,
   `src/squad.py`, `src/match_engine.py`) — 20 fictional clubs, each with
   hidden "true" attack/defense/comeback/fatigue traits that drift slowly
   season to season, simulate 5 seasons (1,900 matches) of shots, goals,
   and red cards minute by minute, including hidden per-match "day form"
   and short in-match "pressure spells" that real-looking broadcast data
   would also contain.
2. **Trait estimation that never cheats** (`src/trait_estimation.py`) — a
   weighted Poisson regression recovers each club's traits purely from
   the generated shot-level data, validated against the known true values
   (r = 0.95–0.98 for attack/defense). A full data-leakage audit
   (`tests/test_home_advantage_and_leakage.py`) confirms the
   win-probability engine never imports a single constant from the
   generator — every number it uses is estimated, the same way it would
   have to be on real data.
3. **A live win-probability + counterfactual engine**
   (`src/win_probability.py`, `src/counterfactual.py`) — a vectorized
   Monte Carlo simulator computes win/draw/loss curves with honest
   uncertainty bands (propagating both simulation noise and trait
   uncertainty), ranks real turning points by total probability swing,
   and re-simulates forward from any one changed event to produce a fair
   forecast-vs-forecast comparison (never comparing a forecast to the
   now-certain real result, which would make every counterfactual look
   more dramatic than it is). Out-of-sample calibration error: 0.0155
   across 2,280 (match, minute) checkpoints — a "70% chance" really wins
   about 70% of the time.
4. **A two-agent AI layer** (`src/agent.py`, `src/agent_tools.py`), the
   newest piece, on **Microsoft Agent Framework**: an **Analyst** agent
   sees the fan's free-text question and four tools wrapping the engine
   above (list events, win probability, run counterfactual, club
   profile), resolves real event minutes/teams before acting, and writes
   an honest `CLARIFY:`/`UNSUPPORTED:` note instead of guessing when a
   question is ambiguous or can't be simulated (e.g. "what if they'd
   subbed earlier?" — no tool for that). A **Narrator** agent, with no
   tools of its own, writes the actual fan-facing reply — in a fan or
   analyst tone, in English or Hebrew — using *only* numbers present in
   the Analyst's tool results. This isn't just prompted; it's
   mechanically checked after every run
   (`src/agent.ungrounded_numbers`), which extracts every number from the
   reply and every tool result and flags any mismatch, working across
   scripts since digits are the same Unicode characters in Hebrew prose
   as in English.
5. **An interactive Streamlit UI** (`app.py`) ties it together: a
   scoreboard, win-probability chart with turning-point cards, a "What
   if...?" panel that runs the counterfactual engine live, and a chat box
   for free-text questions and match recaps.

**Microsoft & Azure technologies**: Microsoft Agent Framework
(`agent-framework` + `agent_framework_openai`'s `OpenAIChatClient`),
Microsoft Foundry / Azure OpenAI (`gpt-5-mini`, Global Standard
deployment), Azure Monitor OpenTelemetry / Application Insights (optional
tracing of every agent run and tool call to the Foundry portal), Azure
Container Apps (hosts the live demo), and Azure AI Speech (neural
voiceover for the demo video only). See the README's "Microsoft & Azure
stack" section for exactly how each is wired in — nothing claimed there
isn't actually used.

**Why it matters.** This is a genuinely new kind of football content: not
a stat, a quantified "what if" with an honest confidence range, explained
conversationally to whichever audience is asking. The same pipeline
(synthetic here, for a fictional league with no licensing constraints)
maps directly onto real match data and a real competition — the
architecture doesn't change, only the data source would.

## Testing instructions for judges

**Fastest path — no setup, ~2 minutes:**

1. Open the live demo: https://sliding-doors.happytree-e6502234.switzerlandnorth.azurecontainerapps.io/
   The default match loads instantly (precomputed).
2. Scroll through the scoreboard, win-probability chart, and the three
   "Top 3 turning points" cards.
3. In "What if...?", pick the 84th-minute goal event (default change
   "Cancel the goal") and click "Run counterfactual" — a live
   re-simulation, a few seconds the first time.
4. In "Ask a what-if question," try:
   - English: *"What if Dunmore had scored their big chance in the 26th
     minute?"* — a grounded, specific answer.
   - Hebrew (switch the language dropdown): *"מה היה קורה אם השער של
     אוקהייבן בדקה ה-90 היה נפסל?"*
   - *"What if they'd made a substitution at halftime?"* — an honest
     refusal (no tool can simulate that), with two concrete alternative
     questions suggested instead.
5. Try "Generate recap" for a plain-language match summary, and the
   "Model power ranking" expander in the sidebar.

**Full path — run it yourself:** follow "Setup & run (from a clean
clone)" in [README.md](README.md) — `git clone`, venv, `pip install -r
requirements.txt`, `streamlit run app.py` (works immediately, no API keys
needed for the core simulation/UI). Add Azure OpenAI credentials to
`.env` to also exercise the AI agent locally. `pytest` runs the full test
suite (55 tests) with no network calls and no cost.

## Prize category fit

- **Best Multi-Agent** (strongest fit) — the Analyst/Narrator split is a
  purposeful two-agent design with a hard boundary: the Analyst can call
  tools but never writes the fan-facing reply, the Narrator writes the
  reply but has no tools and can only use numbers the Analyst's tool
  calls actually returned. The handoff is explicit (tool results +
  analyst note, passed as the Narrator's prompt) and mechanically
  verified (`ungrounded_numbers`), not just described in a docstring.
- **Best Foundry Use** (strong fit) — the agent layer is built entirely
  on Microsoft Agent Framework against a Microsoft Foundry `gpt-5-mini`
  deployment, with deliberate cost/latency tuning
  (`reasoning: {"effort": "low"}`, tuned output-token cap) and optional
  OpenTelemetry tracing of every agent run and tool call straight to the
  Foundry portal.
- **Real-World Impact** — the counterfactual engine and agent
  architecture are domain-agnostic; swapping the synthetic data generator
  for a real match-data feed wouldn't change the architecture, only the
  input.
- Less central fit: **Best Azure Cloud Native** (the app runs on a single
  Azure Container Apps instance — simple and sufficient for a demo, but
  doesn't showcase AKS/Functions/Logic Apps-style cloud-native patterns)
  and **Best Enterprise** (no multi-tenant, auth, or enterprise-scale
  story yet — honest gap, not a focus of the current build).
