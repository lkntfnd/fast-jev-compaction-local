# Local implementation of "fast-jev-compaction" plugin.

Claude Code plugin that replaces the compaction summary with locally hosted
system-one model via LM-Studio / python server. Current best-fit model is a 
[qwen3.5 ]. A good job was made by making Qwen3.5 llm answer in system-one style.
(https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-MLX-bf16)

## What and why

Most context compaction asks an LLM to summarize old turns. A summary is
lossy: a file path, exact error, constraint, or command can disappear even when
it matters later. This library never rewrites anything. It only deletes tool
calls and tool results Jev says are no longer needed, and it asks Jev while
showing it the whole conversation. User and assistant text stays verbatim and
in order.

The repository is both an npm package (`src/`) and a Claude Code plugin
(`hooks/`, `.claude-plugin/`) that uses the package to replace Claude Code's
built-in compaction summary with the original messages.

## How it works

1. Every `tool_use` is paired with its `tool_result` by `tool_use_id`. Calls in
   the first message or in the newest `preserveRecentMessages` messages are
   pinned and never touched.
2. The **state** sent to Jev is the whole conversation so far, oldest first,
   with every tool result replaced by a short note (`ok, 4213 chars (omitted)`).
   Tool inputs are included, texts are included, nothing is summarized.
3. The state is fitted into `maxStateTokens` (25k by default) in stages, each
   applied only if the previous one was not enough: tool inputs truncated to
   1000, then 200, then 60 characters; long texts abridged to head + tail,
   oldest non-pinned messages first; old non-pinned messages collapsed to a
   `[… N chars omitted …]` note; old tool calls reduced to one line each
   (`t12 Read file_path=src/a.ts → ok 480ch`); old call-less messages left
   out; runs of old call-only messages folded into one entry. If it still
   does not fit, compaction throws. Tokens are estimated without a tokenizer (a
   word per six letters, half a token per digit, ~one per other symbol),
   calibrated to land a little above the counts Jev reports.
4. For every non-pinned call Jev gets two `noul` questions: should the **call**
   stay (knowing it was made, with its input, still matters), and should the
   **result** stay verbatim (its contents are still needed and re-running the
   tool would not do).
5. Questions are split into as many requests as needed so state plus questions
   stays under `maxRequestTokens` (30k by default, under Jev's 32k request
   limit). The same full state is resent with every request; requests run
   concurrently and their answers are merged.
6. Decisions per call, against `keepThreshold`:
   - `keepResult ≥ threshold` → keep call and result;
   - else `keepCall ≥ threshold` → keep the call, truncate the result to its
     first `truncateHeadChars` characters plus a one-line note;
   - else → remove the call together with its result.
7. The message list is rebuilt: a message that loses all its content is
   removed, untouched messages are returned as the same objects, and no result
   is ever left without its call.

Jev failures, malformed answers, a missing key, or a history that cannot be
fitted throw; the caller (or the Claude Code hook) decides what to fall back to.
When requests fail, the error is a `PartialCompactionError` whose `result`
applies the answers that did come back (unanswered calls are kept), so a
fallback can start from the partly pruned transcript.

## Install and usage

```sh
npm install fast-jev-compaction
export TYPESAFE_API_KEY=...
```

```ts
import { compactMessages, reductionRatio, type Message } from 'fast-jev-compaction';

const transcript: Message[] = [
  { role: 'user', text: 'Fix the failing test. Never edit src/generated.', toolUses: [] },
  {
    role: 'assistant',
    text: '',
    toolUses: [{ tool_use_id: 'toolu_1', tool: 'Read', input: { file_path: 'src/a.ts' } }],
  },
  { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'toolu_1', text: '…file…' }] },
  // …
];

const result = await compactMessages(transcript, { preserveRecentMessages: 4 });
console.log(result.messages, result.decisions, result.stats);
if (reductionRatio(result) < 0.25) {
  // not worth it: keep the original transcript, or summarize instead
}
```

`Message` is a subset of Claude Code's `SessionMessage`, so a session transcript
can be passed in as is.

To bring your own transport, implement `JevAsker` (one `ask(state, questions)`
method) and call `compact(messages, asker, options)`; `buildJevRequest` and
`parseJevResponse` give you the HTTP request body and response validation.
The building blocks (`collectToolCalls`, `fitState`, `batchCalls`,
`decideCall`, `applyDecisions`) are exported too.

`apiKey` defaults to `process.env.TYPESAFE_API_KEY`. Never commit the key or
put it in a source file.

### Local classifier (LM Studio)

`LocalJevAsker` answers the same questions with a local Jev-style decision
model, [`chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-MLX-bf16`](https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-MLX-bf16),
served by LM Studio. No key is needed and nothing leaves the configured URL.

```ts
import { compact, fetchText, LocalJevAsker } from 'fast-jev-compaction';

const asker = new LocalJevAsker({ fetch: fetchText }); // http://127.0.0.1:1234/v1
const result = await compact(transcript, asker, { preserveRecentMessages: 4 });
```

Every `noul` question becomes one prompt in the model's own format (`[State]`,
`[Question]`, `[Options]` `A. yes` / `B. no`, ending in `Answer:`), sent to
`/chat/completions` with `temperature: 0`, `max_tokens: 2` (LM Studio returns
nothing for 1) and `top_logprobs: 10`. The `noul` is the log-prob of option A
renormalised over A and B; a letter outside the returned candidates gets the
lowest returned log-prob minus 5, as in the model's reference client. The model
never writes text, so `keepThreshold` keeps its meaning.

| Option | Default | Description |
| --- | --- | --- |
| `fetch` | (required) | `fetchText` (global `fetch`) or a host transport |
| `baseUrl` | `http://127.0.0.1:1234/v1` | OpenAI-compatible base URL |
| `model` | `jev-style-qwen3.5-2b-decision-mlx` | Model identifier as LM Studio lists it |
| `concurrency` | `2` | Decisions in flight at once, across every `ask` |
| `contextTokens` | `64000` | Estimated ceiling for one prompt; keep it under the loaded context |

`asker.decide(state, question, options)` answers one question among 2 to 10
mutually exclusive options with the renormalised probability of each and the
most probable `choice`; the Claude Code plugin exposes it to every agent as the
`classify` tool (see [`hooks/README.md`](hooks/README.md)). Pass one
`createScheduler(n)` as `scheduler` to several askers to cap their requests
together.

A prompt over `contextTokens`, an unreachable server, an HTTP error, a response
without log-probs or without any option letter all throw; there is no fallback
to the remote API. The full state is sent once per question, not once per
batch, and LM Studio does not reuse the shared state prefix for this model, so
a local compaction costs one state prefill per question.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `apiKey` | `TYPESAFE_API_KEY` | TypeSafe API key (`compactMessages`/`JevClient`) |
| `model` | `jev-latest` | Jev model name |
| `baseUrl` | `https://api.typesafe.ai/v1/systemone` | System One endpoint |
| `fetch` | native `fetch` | Injectable fetch implementation for tests |
| `goal` | last 3 user prompts | Ongoing task description included in the state |
| `keepThreshold` | `0.5` | Minimum keep probability for a call or result to stay |
| `preserveRecentMessages` | `6` | Newest messages never touched (the first is always kept) |
| `maxStateTokens` | `25000` | Estimated token ceiling for the state |
| `maxRequestTokens` | `30000` | Estimated ceiling for state plus one batch of questions |
| `truncateHeadChars` | `300` | Characters of a dropped tool result retained before its note |

`result.stats` reports message and character counts before and after, the
per-reason decision counts, the state size in estimated tokens, which fitting
stage was needed, and the number of requests.

## Limitations

- Only tool calls and results are candidates; text messages are never removed
  or shortened in the output (they are only abridged in the state Jev sees).
- Token sizes are estimates from character counts, not a tokenizer.
- Calibration is at the request level; a probability is not a proof that a
  result is safe to delete. The assistant can always re-run the tool.
- The full state is repeated with every request, so a history near the state
  ceiling costs one request per handful of questions.

## Claude Code plugin

The repository root is a Claude Code function-hook plugin: `hooks/fast-jev.ts`
is a thin adapter that feeds `session.compact` transcripts through `src/` and
falls back to Claude Code's built-in summary on errors or insufficient
reduction. The `backend` option picks the classifier: `local` (the default,
LM Studio, no key) or `remote` (TypeSafe Jev, needs `TYPESAFE_API_KEY`). See [`hooks/README.md`](hooks/README.md) for configuration and the
Claude Code 2.1.274 type reference.

### Install in Claude Code

Function hooks are an early-access Claude Code feature (2.1.274+), so the
opt-in flag must be set wherever Claude Code runs, e.g. in `~/.claude/settings.json`:

```json
{ "env": { "CLAUDE_CODE_ENABLE_FUNCTION_HOOKS": "1", "TYPESAFE_API_KEY": "<your key>" } }
```

Then add this repository as a plugin marketplace and install the plugin,
either from the shell or as slash commands inside a session:

```sh
claude plugin marketplace add tamaratran/fast-jev-compaction
claude plugin install fast-jev-compaction@fast-jev-compaction
```

The install prompts for the plugin options (backend, API key, thresholds,
`truncateHeadChars`, …); leave them at their defaults to use the local
LM Studio classifier, or set `backend` to `remote` to use `TYPESAFE_API_KEY`
from the environment.
Restart Claude Code or run `/reload-plugins`. From then on `/compact` (and
auto-compaction) goes through Jev: the toast reads
`fast-jev-compaction: kept N/M messages, no summary (…)` when the pruned history
replaced the built-in summary, or `built-in summary over N/M pre-compacted
messages (…)` when Jev could not remove enough (short sessions) or failed
midway: the built-in summary then runs over the transcript already pruned by
the decisions that came back. `turn.complete` asks for compaction once the
context reaches `compactAtPercent` (default 50%).

With the local backend the classification runs in the background once the
context reaches `compactAtPercent`, and the compaction follows a turn or two
later, so no hook waits on the model past Claude Code's 10-second hook budget
(see [`hooks/README.md`](hooks/README.md)).

To run from a checkout without installing: `CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1 claude --plugin-dir .`
from the repository root. No publishing step is required; the marketplace is
just the repo's `.claude-plugin/marketplace.json`.

## Development

```sh
npm install
npm run typecheck        # library + hook
npm test
npm run build
npm run validate:plugin  # claude plugin validate
TYPESAFE_API_KEY="$(cat ~/.typesafe_key)" npm run demo
```

The unit tests use a fake Jev and a fake LM Studio and never contact either. The demo is the live
network check.

## Animated demo (macOS)

`demo/JevDemo` is a small native SwiftUI app that plays a scripted, dramatized
version of the compaction flow inside a Claude Code-style terminal: the tool
calls of a canned transcript are scored, results and calls Jev lets go turn red
and collapse away, and the rest stays verbatim. It never calls the API; it
exists to be screen recorded.

```sh
demo/JevDemo/build.sh   # builds demo/JevDemo/build/JevDemo.app and launches it
```

Press space in the app to replay from the start.
