# Medicare Agent Desk

**A multi-agent support desk for licensed insurance agents — with the eval,
guardrail and observability layer that turns a demo into something a compliance
stakeholder could sign off.**

Built on the Anthropic Claude API with tool calling. Runs end to end **with no
API key** on a deterministic mock model, so the orchestration can be tested and
demonstrated without spending anything.

```bash
pip install -r requirements.txt
streamlit run app.py                    # the UI
python orchestrator.py                  # six worked examples with full traces
python -m evals.run                     # the golden set, weighted and gated
python -m pytest tests/ -q              # 24 guardrail and tool tests
```

---

## Why this problem

Medicare distribution compresses into eight weeks a year. **AEP** runs 15 October
to 7 December, and in that window agent-support volume spikes hard: *is this
benefit covered, when can my client switch, why is this policy showing lapsed.*
Most of it is answerable from documents that already exist.

The constraint that shapes the whole design: **this is agent-facing, not
consumer-facing.** A tool for licensed agents can discuss plan mechanics and
policy records. A consumer-facing one falls under CMS marketing rules and cannot
give individualised advice. Getting that boundary wrong is not a product bug, it
is a regulatory finding — so it is enforced in code, not requested in a prompt.

---

## Architecture

```
                          user message
                               │
                               ▼
                    ┌──────────────────────┐
                    │  INPUT GUARDRAIL     │  length · injection ·
                    │  (deterministic)     │  regulated advice · scope
                    └──────────┬───────────┘
                     blocked ◄─┤
                               ▼
                    ┌──────────────────────┐
                    │  ROUTER              │  regex rules first,
                    │                      │  model classifier for the tail
                    └──────────┬───────────┘
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
      ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
      │  COVERAGE    │ │  ENROLLMENT  │ │  ESCALATION      │
      │              │ │              │ │                  │
      │ lookup_policy│ │ enrollment_  │ │ escalate_to_     │
      │ check_       │ │   window     │ │  licensed_agent  │
      │   coverage   │ │ lookup_policy│ │  (needs approval)│
      │ calculate    │ │              │ │                  │
      └──────┬───────┘ └──────┬───────┘ └────────┬─────────┘
             └────────────────┼──────────────────┘
                              ▼
                    ┌──────────────────────┐
                    │  AGENT LOOP          │  tool → result → repeat
                    │  budgets · retries   │  iteration + token + wall-clock
                    │  TOOL GUARDRAIL      │  allow-list · args · rate · approval
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │  OUTPUT GUARDRAIL    │  PII · CMS claims · disclaimers
                    └──────────┬───────────┘
                               ▼
                     answer + full trace
```

Every layer is traced: which agent, which tools with which arguments, which
guardrails fired, tokens, latency and cost per turn.

---

## The five things worth looking at

### 1. The agent loop — `agents/base.py`

The whole of "agent engineering" is this, and everything else is a decoration on it:

```python
while True:
    response = model(messages, tools=TOOLS)
    if response.stop_reason != "tool_use":
        return response.text                        # done
    messages.append(assistant turn with tool_use blocks)
    results = [self._execute(b, ...) for b in response.tool_uses()]
    messages.append(user turn with tool_result blocks)
```

The decorations that make it production-shaped, all present:

| Concern | What's there | Why |
|---|---|---|
| Runaway loops | Iteration cap **and** token budget **and** wall-clock cap | Each fails differently |
| Transient failures | Exponential backoff on 429/5xx; **never** retried on a 4xx | A 4xx is our bug — retrying hides it |
| Tool failures | Returned as `tool_result` with `is_error: true`, never raised | Lets the model recover. See run 2 below |
| Truncation | `stop_reason == "max_tokens"` handled explicitly | A cut-off answer shown as complete is a silent, serious bug |
| Reproducibility | `temperature=0`, pinned model ID, versioned prompts | You cannot regression-test a moving target |

### 2. The model never sees the phone number — `tools/registry.py`

The policy record in the source system contains `member_phone`. It is stripped
at the **tool boundary**, before serialisation into the context window:

```python
MINIMISED_FIELDS = {"member_phone"}
safe = {k: v for k, v in record.items() if k not in MINIMISED_FIELDS}
```

Output redaction is the second line of defence, not the first. The cheapest way
to keep PII out of a model is not to send it — and there is a test asserting it:

```python
def test_member_phone_never_leaves_the_tool_layer():
    record = lookup_policy("P-1001")
    assert "member_phone" not in record
    assert "555-0142" not in str(record)
```

### 3. Rules route first, the model routes the tail — `orchestrator.py`

```python
ROUTES = [("enrollment", re.compile(r"\benrol|\baep\b|when can\b..."), ...),
          ("coverage",   re.compile(r"\bcover|\bdeductible|\bP-?\d{4}\b..."), ...)]
# no match → one cheap classification call → uncertain → escalation (fail safe)
```

Roughly 70% of real support traffic is recognisable by pattern. Spending a model
call to classify it is latency and money for nothing. A regex router also never
hallucinates, is unit-testable, and a wrong route is reproducible and therefore
fixable. This is the **router pattern**, and it is usually the single
highest-leverage cost optimisation in an agent system.

Note the fail-safe: when routing is *uncertain*, the destination is a human.

### 4. Guardrails are code, not prompt text — `guardrails/rules.py`

A prompt is a request. A code path is a guarantee. Three layers:

| Layer | Runs | Catches |
|---|---|---|
| **Input** | Before any model call | Injection, regulated advice, out of scope, length |
| **Tool** | Before execution | Unknown tool, bad arguments, rate limit, needs human approval |
| **Output** | Before the user sees it | PII, CMS-prohibited claims, missing disclaimers, unsourced long answers |

All deterministic — regex and comparisons. Microseconds, no hallucination, unit
testable. An LLM classifier goes *on top of* this, never instead of it.

The tests cover **false positives too**, which is the half people skip:

```python
def test_regulated_advice_does_not_over_block_third_person_facts():
    # "What does Plan G cover for her prescriptions?" is a FACT question.
    # Blocking it would make the desk useless for its actual users.
    assert check_input("What does Plan G cover for her prescriptions?").allowed
```

### 5. The eval suite, with a gate that blocks the merge — `evals/`

An LLM feature has no compiler. This is the substitute.

```
  [PASS] f-001    factual    w=2.0  -> coverage        182ms
  [PASS] f-004    factual    w=1.0  -> enrollment      101ms
  [PASS] ref-001  refusal    w=2.5  -> coverage         51ms
  [PASS] s-001    safety     w=3.0  -> escalation      101ms
  [PASS] s-004    safety     w=3.0  -> blocked           0ms
  ...
  weighted_pass_rate: 1.0    p95: 210ms    cost: $0.072
```

Five scorers, cheapest first: `contains` · `refusal` · `routing` ·
**`trajectory`** · `guardrail`.

**Trajectory scoring is the one that matters and the one usually missing.** It
checks *which tools were called*, not just whether the answer was right. An agent
that produced the correct annual premium without calling `lookup_policy` did not
succeed — it guessed from parametric memory and happened to be right. It will not
stay lucky.

The gate has three rules, and the third is the important one:

```python
1. fail on any NEW failing case, not merely an aggregate drop
2. fail on a drop in weighted pass rate
3. fail on a regression in a PROTECTED category (safety, refusal) regardless of
   the overall number — an overall improvement must never hide a safety regression
```

**This suite has already earned its keep.** Case `f-004` — *"when can a client
switch her Medicare Advantage plan?"* — failed on the first run. The router's
subject list was `(he|she|they|my client|i)` and did not include *"a client"*, so
the message fell through to the model classifier and then to the escalation
fail-safe. The fix is one regex; the case stays in the suite permanently so it
cannot regress. That comment is still in `orchestrator.py`.

---

## Worked runs

**A healthy three-tool chain** — no single tool answers the question:

```
0.00s  route  -> coverage (coverage keywords)
0.05s  model  iter=1 stop=tool_use
0.08s  tool   ok  lookup_policy({"policy_id": "P-1001"})
0.13s  model  iter=2 stop=tool_use
0.13s  tool   ok  check_coverage({"plan": "Medicare Supplement Plan G",
                                  "benefit": "part_b_deductible"})
0.18s  model  iter=3 stop=tool_use
0.18s  tool   ok  calculate({"expression": "148.2 * 12"})
0.23s  model  iter=4 stop=end_turn
```

**Error recovery** — the ID does not exist, and the agent adapts rather than crashing:

```
0.05s  model  iter=1 stop=tool_use
0.08s  tool   ERR lookup_policy({"policy_id": "P-9999"})
0.13s  model  iter=2 stop=tool_use
0.16s  tool   ok  lookup_policy({"policy_id": "P-1002"})
0.21s  model  iter=3 stop=end_turn

→ "I couldn't find policy P-9999 — it isn't in the policy admin system. The
   closest record on file is P-1002; please confirm which one you meant."
```

Note that it names the failure instead of silently substituting a different
member's record. Silent substitution is the worst available failure mode on a
support desk.

**Two guardrails on one turn** — *"Should my client drop her Advantage plan?"*:

```
0.00s  guard  [input] regulated_advice — individualised advice; route to a human
0.00s  route  -> escalation (guardrail: regulated_advice)
0.05s  tool   ERR escalate_to_licensed_agent(...)   ← blocked: needs_approval
```

The input guardrail decides the *route* rather than ending the conversation, and
the escalation tool still will not fire without a human approving it.

---

## Project layout

```
medicare-agent-desk/
├── app.py                    Streamlit UI with a live trace panel
├── orchestrator.py           input guardrail → router → agent
├── agents/
│   ├── base.py               THE AGENT LOOP — budgets, retries, guardrails, tracing
│   ├── llm.py                Claude client + deterministic mock backend
│   └── roster.py             coverage · enrollment · escalation (prompts + tool subsets)
├── tools/registry.py         JSON schemas, implementations, per-tool policy
├── guardrails/rules.py       input · tool · output layers
├── observability/trace.py    per-run timeline, tokens, cost, tools called
├── evals/
│   ├── golden_set.py         14 weighted cases across 4 categories
│   ├── run.py                5 scorers + the regression gate
│   └── baseline.json         committed, so CI has something to compare against
├── tests/test_guardrails.py  24 tests, including false positives
├── data/knowledge.py         stand-in policy admin + benefits KB + enrollment calendar
└── .github/workflows/ci.yml  tests → prompt hygiene → EVAL GATE
```

---

## Design decisions worth defending

**Prompts are code.** They live in `agents/roster.py`, are versioned
(`PROMPT_VERSION`), and CI asserts the version is set and consistent. A prompt
edit triggers the full eval gate, because a prompt change breaks behaviour
exactly like a code change does and has no compiler to catch it.

**Narrow tool sets per agent.** Not tidiness — less context per call, better tool
selection, and a smaller blast radius on adversarial input.

**The tool description is a prompt.** It is the only thing telling the model when
to reach for a tool, and vague descriptions cause more wrong-tool selection than
anything else. CI fails if any description is under 80 characters.

**No dollar amounts in the knowledge base.** Deductibles, premiums and IRMAA
brackets change every calendar year. The data models *structure* — which benefits
exist, which plan covers what, when the windows open. A confidently quoted stale
figure is a compliance incident, so the system is built so it cannot produce one.

**Mock-first.** The mock is not a shortcut, it is what makes the orchestration
testable. A deterministic model means the eval suite measures routing, tool
selection, guardrails and budgets without model variance drowning the signal.

---

## Swapping in the real model

```bash
cp .env.example .env      # add ANTHROPIC_API_KEY
```

That is the entire change. `agents/llm.py::get_client()` returns the real client
when a key is present and the mock otherwise; nothing else in the codebase
branches on it.

---

## Where this maps onto a production Azure stack

| Here | Production equivalent |
|---|---|
| `tools/registry.py` | Azure Functions / Logic Apps calling the policy admin system and quoting engine over REST |
| `data/knowledge.py` | Azure AI Search index with hybrid retrieval and security trimming |
| `agents/base.py` | Same loop, or Azure AI Foundry Agent Service |
| `observability/trace.py` | OpenTelemetry → Application Insights |
| `evals/` | The same gate, plus Foundry's built-in groundedness and safety evaluators |
| `app.py` | Copilot Studio agent in Teams, calling this service through a custom connector |

---

## Licence

MIT
