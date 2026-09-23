import type {
  Hook,
  On,
  PluginOptions,
  Register,
  SessionMessage,
  ToolResultSummary,
  ToolUseSummary,
  TurnCompleteInput,
} from 'claude-code';

import {
  classifyCalls,
  compact,
  compactWithAnswers,
  PartialCompactionError,
  reductionRatio,
  resolveOptions,
  type Classification,
} from '../src/compact.js';
import {
  LOCAL_BASE_URL,
  LOCAL_CONCURRENCY,
  LOCAL_CONTEXT_TOKENS,
  LOCAL_MODEL,
  LocalJevAsker,
  MAX_OPTIONS,
  createScheduler,
  type LocalJevAskerOptions,
  type Scheduler,
} from '../src/local.js';
import { buildJevRequest, DEFAULT_MODEL, parseJevResponse } from '../src/request.js';
import type {
  CompactOptions,
  CompactResult,
  JevAsker,
  Message,
  ToolResult,
  ToolUse,
} from '../src/types.js';

const HOOK_DEFAULTS = {
  backend: 'local',
  compactAtPercent: 50,
  minReductionRatio: 0.25,
  model: DEFAULT_MODEL,
  localBaseUrl: LOCAL_BASE_URL,
  localModel: LOCAL_MODEL,
  localConcurrency: LOCAL_CONCURRENCY,
  localContextTokens: LOCAL_CONTEXT_TOKENS,
  classifyTool: true,
  localDeadlineMs: 7000,
};

export type HookFetchInit = {
  method?: string;
  headers?: Record<string, string>;
  body?: string;
};

export type HookFetchResponse = {
  status: number;
  ok: boolean;
  text: string;
};

/** The shape of `$.http.fetch`, so the hook can be driven without an engine. */
export type HookFetch = (url: string, init?: HookFetchInit) => Promise<HookFetchResponse>;

export type HookConfig = CompactOptions & {
  /** `local` (LM Studio) or `remote` (TypeSafe Jev); anything else fails the compaction. */
  backend: string;
  apiKey?: string;
  compactAtPercent: number;
  minReductionRatio: number;
  model: string;
  localBaseUrl: string;
  localModel: string;
  localConcurrency: number;
  localContextTokens: number;
  /** Registers the `classify` tool for every agent in the session. */
  classifyTool: boolean;
  /**
   * Longest a hook waits on the local model before answering with what it
   * has; Claude Code drops a hook that runs past 10 s.
   */
  localDeadlineMs: number;
};

function optionNumber(options: PluginOptions, key: string, fallback: number): number {
  const value = options[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function optionBoolean(options: PluginOptions, key: string, fallback: boolean): boolean {
  const value = options[key];
  return typeof value === 'boolean' ? value : fallback;
}

function optionString(options: PluginOptions, key: string): string | undefined {
  const value = options[key];
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

/** Reads the plugin's `userConfig` values; anything missing takes the defaults. */
export function resolveHookConfig(options: PluginOptions): HookConfig {
  const numbers: Partial<Omit<CompactOptions, 'goal'>> = {};
  for (const key of [
    'keepThreshold',
    'preserveRecentMessages',
    'maxStateTokens',
    'maxRequestTokens',
    'truncateHeadChars',
  ] as const) {
    const value = options[key];
    if (typeof value === 'number' && Number.isFinite(value)) numbers[key] = value;
  }
  const config: HookConfig = {
    ...numbers,
    backend: (optionString(options, 'backend') ?? HOOK_DEFAULTS.backend).trim().toLowerCase(),
    compactAtPercent: optionNumber(options, 'compactAtPercent', HOOK_DEFAULTS.compactAtPercent),
    minReductionRatio: optionNumber(
      options,
      'minReductionRatio',
      HOOK_DEFAULTS.minReductionRatio,
    ),
    model: optionString(options, 'model') ?? HOOK_DEFAULTS.model,
    localBaseUrl: optionString(options, 'localBaseUrl') ?? HOOK_DEFAULTS.localBaseUrl,
    localModel: optionString(options, 'localModel') ?? HOOK_DEFAULTS.localModel,
    localConcurrency: optionNumber(options, 'localConcurrency', HOOK_DEFAULTS.localConcurrency),
    localContextTokens: optionNumber(
      options,
      'localContextTokens',
      HOOK_DEFAULTS.localContextTokens,
    ),
    classifyTool: optionBoolean(options, 'classifyTool', HOOK_DEFAULTS.classifyTool),
    localDeadlineMs: optionNumber(options, 'localDeadlineMs', HOOK_DEFAULTS.localDeadlineMs),
  };
  const apiKey = optionString(options, 'apiKey');
  if (apiKey) config.apiKey = apiKey;
  const goal = optionString(options, 'goal');
  if (goal) config.goal = goal;
  return config;
}

/** A `JevAsker` over the engine's `$.http.fetch`. */
export function jevAsker(fetchFn: HookFetch, apiKey: string, model: string): JevAsker {
  return {
    async ask(state, questions) {
      const request = buildJevRequest({ apiKey, model }, state, questions);
      const response = await fetchFn(request.url, {
        method: request.method,
        headers: request.headers,
        body: request.body,
      });
      return parseJevResponse(response.status, response.ok, response.text);
    },
  };
}

function toolUseSummary(tool: ToolUse): ToolUseSummary {
  const summary: ToolUseSummary = {
    tool_use_id: tool.tool_use_id,
    tool: tool.tool,
    input: tool.input,
  };
  if (tool.text !== undefined) summary.text = tool.text;
  if (tool.isError) summary.isError = true;
  return summary;
}

function toolResultSummary(result: ToolResult): ToolResultSummary {
  return {
    tool_use_id: result.tool_use_id,
    text: result.text,
    isError: result.isError ?? false,
  };
}

/**
 * Maps the library's output back onto session messages. Whatever came back
 * unchanged (a message, a tool use, a tool result) is the engine's own object,
 * handle included; anything rebuilt is a fresh message without a handle, so the
 * engine takes the edited content instead of its original.
 */
export function toSessionMessages(
  input: readonly SessionMessage[],
  output: readonly Message[],
): SessionMessage[] {
  const messages = new Map<Message, SessionMessage>();
  const uses = new Map<ToolUse, ToolUseSummary>();
  const results = new Map<ToolResult, ToolResultSummary>();
  for (const message of input) {
    messages.set(message, message);
    for (const tool of message.toolUses) uses.set(tool, tool);
    for (const result of message.toolResults ?? []) results.set(result, result);
  }
  return output.map((message) => {
    const own = messages.get(message);
    if (own) return own;
    const rebuilt: SessionMessage = {
      role: message.role,
      text: message.text,
      toolUses: message.toolUses.map((tool) => uses.get(tool) ?? toolUseSummary(tool)),
    };
    if (message.toolResults && message.toolResults.length > 0) {
      rebuilt.toolResults = message.toolResults.map(
        (result) => results.get(result) ?? toolResultSummary(result),
      );
    }
    return rebuilt;
  });
}

export type SessionCompaction = {
  result: CompactResult;
  messages: SessionMessage[];
};

/** One line naming the classifier a compaction will use; never any transcript content. */
export function classifierLabel(config: HookConfig): string {
  if (config.backend === 'local') {
    let endpoint = config.localBaseUrl;
    try {
      endpoint = new URL(config.localBaseUrl).host;
    } catch {
      // Shown as configured; the request itself reports the bad URL.
    }
    return `classifier=local model=${config.localModel} endpoint=${endpoint} concurrency=${config.localConcurrency}`;
  }
  return `classifier=remote model=${config.model}`;
}

/** The LM Studio asker from the `local*` options, whatever `backend` is. */
export function localAsker(
  config: HookConfig,
  fetchFn: HookFetch,
  scheduler?: Scheduler,
  extra: Pick<LocalJevAskerOptions, 'priority' | 'deadline' | 'signal'> = {},
): LocalJevAsker {
  return new LocalJevAsker({
    baseUrl: config.localBaseUrl,
    model: config.localModel,
    concurrency: config.localConcurrency,
    contextTokens: config.localContextTokens,
    fetch: fetchFn,
    ...(scheduler ? { scheduler } : {}),
    ...extra,
  });
}

/**
 * The asker for the configured backend. `local` only ever talks to
 * `localBaseUrl` and needs no key; `remote` needs the TypeSafe key.
 */
export function classifierFor(
  config: HookConfig,
  fetchFn: HookFetch,
  scheduler?: Scheduler,
): JevAsker {
  if (config.backend === 'local') return localAsker(config, fetchFn, scheduler);
  if (config.backend !== 'remote') {
    throw new Error(`unknown backend ${JSON.stringify(config.backend)} (use local or remote)`);
  }
  if (!config.apiKey) throw new Error('TYPESAFE_API_KEY is not configured');
  return jevAsker(fetchFn, config.apiKey, config.model);
}

/** Runs the library over a session transcript; throws when the classifier cannot answer. */
export async function compactSession(
  messages: readonly SessionMessage[],
  config: HookConfig,
  fetchFn: HookFetch,
  scheduler?: Scheduler,
): Promise<SessionCompaction> {
  const result = await compact(messages, classifierFor(config, fetchFn, scheduler), config);
  return { result, messages: toSessionMessages(messages, result.messages) };
}

function percent(ratio: number): string {
  return `${Math.round(ratio * 100)}%`;
}

export function summarize(result: CompactResult): string {
  const { stats } = result;
  const parts = [
    stats.kept > 0 ? `${stats.kept} kept` : '',
    stats.resultsDropped > 0 ? `${stats.resultsDropped} results truncated` : '',
    stats.callsDropped > 0 ? `${stats.callsDropped} call_dropped` : '',
    stats.pinned > 0 ? `${stats.pinned} pinned` : '',
  ].filter(Boolean);
  return `${percent(reductionRatio(result))} reduction; ${
    parts.join(', ') || 'no tool calls'
  }; state ~${stats.stateTokens} tokens (${stats.stateStage}) in ${stats.requests} request(s)`;
}

const UI_LOG_MAX_CHARS = 4096;

export function decisionLog(result: CompactResult): string {
  return result.decisions
    .filter((d) => d.reason !== 'pinned')
    .map(
      (d) =>
        `${d.id}:${d.tool}:${d.action}/call=${d.keepCall.toFixed(2)}/result=${d.keepResult.toFixed(2)}`,
    )
    .join(' ');
}

export function decisionLogLines(
  result: CompactResult,
  maxChars: number = UI_LOG_MAX_CHARS,
): string[] {
  const entries = decisionLog(result).split(' ').filter(Boolean);
  if (entries.length === 0) return ['decisions: (none)'];
  const chunks: string[] = [];
  let current = '';
  for (const entry of entries) {
    const next = current ? `${current} ${entry}` : entry;
    if (current && next.length > maxChars - 24) {
      chunks.push(current);
      current = entry;
    } else current = next;
  }
  chunks.push(current);
  return chunks.map((chunk, index) =>
    chunks.length === 1
      ? `decisions: ${chunk}`
      : `decisions (${index + 1}/${chunks.length}): ${chunk}`,
  );
}

/** `$.http.fetch` as a `HookFetch`. */
function hostFetch($: {
  http: {
    fetch: (
      url: string,
      init?: HookFetchInit,
    ) => Promise<{ status: number; ok: boolean; text: string }>;
  };
}): HookFetch {
  return async (url, init) => {
    const response = await $.http.fetch(url, init);
    return { status: response.status, ok: response.ok, text: response.text };
  };
}

async function getApiKey(
  $: {
    env: { get: (name: string) => Promise<string | undefined> };
    settings: { read: () => Promise<Readonly<Record<string, unknown>>> };
  },
  config: HookConfig,
): Promise<string | undefined> {
  if (config.apiKey) return config.apiKey;
  const fromEnv = await $.env.get('TYPESAFE_API_KEY');
  if (fromEnv) return fromEnv;
  const settings = await $.settings.read();
  const env = settings['env'];
  if (env && typeof env === 'object') {
    const value = (env as Record<string, unknown>)['TYPESAFE_API_KEY'];
    if (typeof value === 'string' && value) return value;
  }
  return undefined;
}

function notify(
  $: {
    ui: {
      log: (text: string) => void;
      toast: (text: string, options?: { timeoutMs?: number }) => void;
    };
  },
  text: string,
): void {
  $.ui.log(text);
  $.ui.toast(text, { timeoutMs: 15_000 });
}

export const CLASSIFY_DESCRIPTION = `Fast local decision classifier (a Jev-style model on this machine). It reads \`state\`, answers \`question\` by choosing one of \`options\`, and returns calibrated probabilities as JSON: {"choice": "<option>", "probabilities": {"<option>": p, ...}}, in option order, summing to 1.
It cannot read files, run commands or see the conversation: it judges only the text in \`state\`. Put the actual evidence there (the relevant diff, test output, requirements, error text), not claims such as "I implemented it well".
Options must be mutually exclusive, 2 to ${MAX_OPTIONS} of them.
Compare the probability of the option you care about with a threshold (for example act only when it is at least 0.8) instead of trusting \`choice\` alone.
It weighs explicit outcomes (failing tests, error messages) well but does not reliably spot problems hidden in raw code, such as injection bugs, or subtle criteria gaps: state findings explicitly (test, linter, scanner output).
It is a triage/gate signal, not a substitute for running tests or a full review, and never the only security check.
Examples:
- "Is this implementation acceptable to merge?" options ["acceptable", "needs changes"]
- "Is this change free of security problems?" options ["secure", "has a security issue"]
- "Does the output meet the acceptance criteria?" options ["meets all criteria", "misses some criteria"]`;

export const CLASSIFY_INPUT_SCHEMA = {
  type: 'object',
  properties: {
    state: {
      type: 'string',
      description:
        'The evidence the decision rests on: the relevant diff, test output, requirements, error text.',
    },
    question: { type: 'string', description: 'One question, answered by choosing one option.' },
    options: {
      type: 'array',
      items: { type: 'string', minLength: 1 },
      minItems: 2,
      maxItems: MAX_OPTIONS,
      uniqueItems: true,
      description: `2 to ${MAX_OPTIONS} mutually exclusive answers.`,
    },
  },
  required: ['state', 'question', 'options'],
  additionalProperties: false,
} as const;

export type ClassifyOutcome =
  | { result: string; log: string }
  | { deny: string; log: string };

/**
 * Serves one `classify` call: validates the input, asks the local model and
 * formats the answer. Never throws; every failure is a `deny` the calling
 * agent reads as the tool's error. The log line carries no state or question.
 */
export async function classifyCall(
  input: Readonly<Record<string, unknown>>,
  asker: LocalJevAsker,
): Promise<ClassifyOutcome> {
  const started = Date.now();
  const { state, question, options } = input;
  const optionCount = Array.isArray(options) ? options.length : 0;
  const fail = (reason: string): ClassifyOutcome => ({
    deny: `classify failed: ${reason}`,
    log: `classify: failed options=${optionCount} ms=${Date.now() - started} (${reason.slice(0, 200)})`,
  });
  if (typeof state !== 'string' || state.trim().length === 0) {
    return fail('state must be a non-empty string');
  }
  if (typeof question !== 'string' || question.trim().length === 0) {
    return fail('question must be a non-empty string');
  }
  try {
    const decision = await asker.decide(state, question, options as readonly string[]);
    const declared = options as readonly string[];
    // Built by hand so the probabilities keep option order even for numeric-looking options.
    const probabilities = declared
      .map((option, index) => `${JSON.stringify(option)}: ${decision.probabilities[index]}`)
      .join(', ');
    return {
      result: `{"choice": ${JSON.stringify(decision.choice)}, "probabilities": {${probabilities}}}`,
      log: `classify: options=${declared.length} prompt_tokens=${decision.promptTokens ?? '?'} ms=${
        Date.now() - started
      } choice=${JSON.stringify(decision.choice.slice(0, 40))} p=${decision.probabilities[
        declared.indexOf(decision.choice)
      ]!.toFixed(3)}`,
    };
  } catch (error) {
    return fail(error instanceof Error ? error.message : String(error));
  }
}

export const register: Register = (on: On, options: PluginOptions) => {
  const configured = resolveHookConfig(options);
  // One queue for every request to the local model: compactions and classify calls alike.
  const scheduler = createScheduler(configured.localConcurrency);
  let compacting = false;
  let classifyTool: string | undefined;

  on('session.start', async ($, event, next) => {
    if (configured.classifyTool) {
      try {
        const { tool } = await $.tool.register({
          name: 'classify',
          description: CLASSIFY_DESCRIPTION,
          inputSchema: CLASSIFY_INPUT_SCHEMA,
        });
        classifyTool = tool;
        $.ui.log(`classify tool registered as ${tool} (model=${configured.localModel})`);
      } catch (error) {
        $.ui.log(
          `classify tool not registered (${error instanceof Error ? error.message : String(error)})`,
        );
      }
    }
    return next(event);
  });

  on('tool.call', { tool: /__classify$/ }, async ($, event, next) => {
    if (classifyTool === undefined || event.tool !== classifyTool) return next(event);
    const asker = localAsker(configured, hostFetch($), scheduler, {
      deadline: Date.now() + configured.localDeadlineMs,
    });
    const outcome = await classifyCall(event as Readonly<Record<string, unknown>>, asker);
    $.ui.log(outcome.log);
    return 'deny' in outcome ? { deny: outcome.deny } : { result: outcome.result };
  });

  // What this session's hooks share: the background classification of the
  // main transcript, started once it reached compactAtPercent.
  const session: SessionState = {};

  on('session.compact', async ($, event, next) => {
    if (event.agentId !== undefined) {
      $.ui.log(`subagent ${event.agentId} compaction left to the built-in summary`);
      return next(event);
    }
    const started = Date.now();
    const job = session.precomputed;
    session.precomputed = undefined;
    try {
      if (configured.backend !== 'local') {
        const config = { ...configured, apiKey: await getApiKey($, configured) };
        $.ui.log(classifierLabel(config));
        const { result } = await compactSession(event.messages, config, hostFetch($), scheduler);
        return settle($, event, next, result, undefined, configured);
      }
      $.ui.log(classifierLabel(configured));
      let classification: Classification;
      if (job?.done && job.classification) {
        classification = job.classification;
        $.ui.log(
          `compacting with background answers from ${Math.round((started - job.started) / 1000)}s ago`,
        );
      } else {
        job?.controller.abort();
        const asker = localAsker(configured, hostFetch($), scheduler, {
          deadline: started + configured.localDeadlineMs,
        });
        classification = await classifyCalls(event.messages, asker, configured);
        $.ui.log(
          `compacting with answers from now${
            classification.timedOut
              ? ` (time budget reached: ${classification.answers.size}/${classification.candidates} calls decided, the rest kept)`
              : ''
          }`,
        );
      }
      const result = compactWithAnswers(event.messages, classification, configured, started);
      const [failure] = classification.failures;
      return settle(
        $,
        event,
        next,
        result,
        failure === undefined
          ? undefined
          : `${errorText(failure)} (${classification.answers.size}/${classification.candidates} calls decided)`,
        configured,
      );
    } catch (error) {
      if (error instanceof PartialCompactionError) {
        return settle($, event, next, error.result, errorText(error), configured);
      }
      notify($, `fallback to built-in summary (${errorText(error)})`);
      return next(event);
    }
  });

  on('turn.complete', async ($, event: TurnCompleteInput, next) => {
    // Only the main loop's turns can trigger auto-compaction.
    if (compacting || event.agentId !== undefined) return next(event);
    try {
      const { context } = await $.session.usage();
      if ((context.percent ?? 0) < configured.compactAtPercent) {
        // Compacted or cleared meanwhile: whatever was classified is stale.
        if (session.precomputed) {
          session.precomputed.controller.abort();
          session.precomputed = undefined;
        }
        return next(event);
      }
      if (configured.backend === 'local' && !session.precomputed?.done) {
        if (!session.precomputed) await startPrecomputation($, session, configured, scheduler);
        return next(event);
      }
      compacting = true;
      // Not awaited: a built-in summary takes longer than this hook may run.
      $.session
        .compact()
        .catch((error: unknown) => $.ui.log(`auto-compact failed (${errorText(error)})`))
        .finally(() => {
          compacting = false;
        });
    } catch (error) {
      $.ui.log(`auto-compact skipped (${errorText(error)})`);
    }
    return next(event);
  });
};

type SessionState = { precomputed?: Precomputation };

/**
 * Classifies the main transcript in the background, at low priority in the
 * shared queue, so a later compaction applies the answers without waiting.
 */
async function startPrecomputation(
  $: HookHost,
  session: SessionState,
  configured: HookConfig,
  scheduler: Scheduler,
): Promise<void> {
  const messages = await $.session.messages();
  const controller = new AbortController();
  const job: Precomputation = { done: false, controller, started: Date.now() };
  session.precomputed = job;
  const asker = localAsker(configured, hostFetch($), scheduler, {
    priority: 'low',
    signal: controller.signal,
  });
  $.ui.log(`background classification started over ${messages.length} messages`);
  classifyCalls(messages, asker, configured)
    .then((classification) => {
      job.classification = classification;
      if (session.precomputed !== job) return;
      const [failure] = classification.failures;
      $.ui.log(
        `background classification done: ${classification.answers.size}/${classification.candidates} calls in ${Math.round((Date.now() - job.started) / 1000)}s${
          failure === undefined ? '' : ` (${errorText(failure)})`
        }`,
      );
    })
    .catch((error: unknown) => {
      if (session.precomputed === job) {
        $.ui.log(`background classification failed (${errorText(error)})`);
      }
    })
    .finally(() => {
      job.done = true;
    });
}

/**
 * Answers `session.compact` from a result: the pruned history itself when
 * nothing failed and it saves enough, otherwise the built-in summary over the
 * pruned history.
 */
async function settle(
  $: HookHost,
  event: CompactEvent,
  next: CompactNext,
  result: CompactResult,
  failure: string | undefined,
  configured: HookConfig,
) {
  const messages = toSessionMessages(event.messages, result.messages);
  for (const line of decisionLogLines(result)) $.ui.log(line);
  if (failure !== undefined) {
    // The decisions that did come back still prune what the summarizer reads.
    notify(
      $,
      `built-in summary over ${messages.length}/${event.messages.length} partially pre-compacted messages (${failure}; ${summarize(result)})`,
    );
    return next({ ...event, messages });
  }
  if (reductionRatio(result) < configured.minReductionRatio) {
    notify(
      $,
      `built-in summary over ${messages.length}/${event.messages.length} pre-compacted messages (below ${percent(configured.minReductionRatio)} minimum: ${summarize(result)})`,
    );
    return next({ ...event, messages });
  }
  notify(
    $,
    `kept ${messages.length}/${event.messages.length} messages, no summary (${summarize(result)})`,
  );
  return { messages };
}

type Precomputation = {
  done: boolean;
  controller: AbortController;
  started: number;
  classification?: Classification;
};

type HookHost = Parameters<Hook<'session.compact'>>[0];
type CompactEvent = Parameters<Hook<'session.compact'>>[1];
type CompactNext = Parameters<Hook<'session.compact'>>[2];

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export { resolveOptions };
