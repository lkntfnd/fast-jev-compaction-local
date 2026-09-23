import { describe, expect, it } from 'vitest';
import {
  classifierLabel,
  compactSession,
  register,
  decisionLog,
  decisionLogLines,
  resolveHookConfig,
  summarize,
  toSessionMessages,
} from '../hooks/fast-jev.ts';
import { applyDecisions, collectToolCalls, decideCall, type Message } from '../src/index.js';

type SessionMessage = Message & { handle?: string };

function message(role: Message['role'], text: string, extra: Partial<SessionMessage> = {}): SessionMessage {
  return { role, text, toolUses: [], ...extra };
}

function call(id: string, tool: string, input: Record<string, unknown>, text: string): SessionMessage {
  return message('assistant', '', {
    toolUses: [{ tool_use_id: id, tool, input, text }],
    handle: `h-${id}`,
  });
}

function result(id: string, text: string, isError = false): SessionMessage {
  return message('user', '', { toolResults: [{ tool_use_id: id, text, isError }], handle: `r-${id}` });
}

const fileA = 'export const a = 1;\n'.repeat(50);

function transcript(): SessionMessage[] {
  return [
    message('user', 'Fix the failing test.', { handle: 'h-0' }),
    call('tool-1', 'Read', { file_path: 'src/a.ts' }, fileA),
    result('tool-1', fileA),
    call('tool-2', 'Bash', { command: 'npm test' }, 'FAIL'),
    result('tool-2', 'FAIL b.test.ts: expected 2 to be 3', true),
    message('assistant', 'Fixing now.', { handle: 'h-5' }),
    message('user', 'go ahead', { handle: 'h-6' }),
  ];
}

function jevFetch(answer: (name: string) => number, bodies: string[] = []) {
  return async (_url: string, init?: { body?: string }) => {
    bodies.push(init?.body ?? '');
    const { questions } = JSON.parse(init?.body ?? '{}') as { questions: Record<string, unknown> };
    const answers = Object.fromEntries(
      Object.keys(questions).map((key) => [key, { type: 'noul', noul: answer(key) }]),
    );
    return { status: 200, ok: true, text: JSON.stringify({ answers }) };
  };
}

const localDefaults = {
  localBaseUrl: 'http://127.0.0.1:1234/v1',
  localModel: 'jev-style-qwen3.5-2b-decision-mlx',
  localConcurrency: 2,
  localContextTokens: 64000,
  classifyTool: true,
  localDeadlineMs: 7000,
};

function lmStudioFetch(answer: (prompt: string) => number, urls: string[] = []) {
  return async (url: string, init?: { body?: string }) => {
    urls.push(url);
    const prompt = (JSON.parse(init?.body ?? '{}') as { messages: Array<{ content: string }> }).messages[0]!.content;
    const p = answer(prompt);
    const top = [
      { token: ' A', logprob: Math.log(p) },
      { token: ' B', logprob: Math.log(1 - p) },
    ];
    return {
      status: 200,
      ok: true,
      text: JSON.stringify({ choices: [{ logprobs: { content: [{ top_logprobs: top }] } }], usage: { prompt_tokens: 10 } }),
    };
  };
}

describe('hook config', () => {
  it('reads userConfig values and falls back to defaults', () => {
    expect(resolveHookConfig({})).toEqual({
      backend: 'local',
      compactAtPercent: 50,
      minReductionRatio: 0.25,
      model: 'jev-latest',
      ...localDefaults,
    });
    expect(
      resolveHookConfig({ apiKey: 'k', keepThreshold: 0.3, maxStateTokens: 1000, model: 'jev-x', goal: 'g', compactAtPercent: 'no' }),
    ).toEqual({
      backend: 'local',
      apiKey: 'k',
      keepThreshold: 0.3,
      maxStateTokens: 1000,
      model: 'jev-x',
      goal: 'g',
      compactAtPercent: 50,
      minReductionRatio: 0.25,
      ...localDefaults,
    });
    expect(
      resolveHookConfig({
        backend: ' Remote ',
        localBaseUrl: 'http://localhost:8080/v1',
        localModel: 'm',
        localConcurrency: 4,
        localContextTokens: 32000,
      }),
    ).toMatchObject({
      backend: 'remote',
      localBaseUrl: 'http://localhost:8080/v1',
      localModel: 'm',
      localConcurrency: 4,
      localContextTokens: 32000,
    });
  });

  it('names the classifier without any transcript content', () => {
    expect(classifierLabel(resolveHookConfig({}))).toBe(
      'classifier=local model=jev-style-qwen3.5-2b-decision-mlx endpoint=127.0.0.1:1234 concurrency=2',
    );
    expect(classifierLabel(resolveHookConfig({ backend: 'remote', model: 'jev-x' }))).toBe(
      'classifier=remote model=jev-x',
    );
  });
});

describe('session message mapping', () => {
  it('returns the engine objects for untouched messages and handle-less copies for rebuilt ones', () => {
    const messages = transcript();
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 }),
      decideCall(calls[1]!, { keepCall: 0.9, keepResult: 0.9 }, { keepThreshold: 0.5 }),
    ];
    messages[1]!.toolUses[0]!.text = 'x'.repeat(2000);
    messages[2]!.toolResults![0]!.text = 'x'.repeat(2000);
    const out = toSessionMessages(messages, applyDecisions(messages, decisions, calls, 300));
    expect(out).toHaveLength(messages.length);
    expect(out[0]).toBe(messages[0]);
    expect(out[1]?.handle).toBeUndefined();
    expect(out[1]?.toolUses[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[fast-jev-compaction truncated 1700 chars`),
    );
    expect(out[2]?.handle).toBeUndefined();
    expect(out[2]?.toolResults?.[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[fast-jev-compaction truncated 1700 chars`),
    );
    expect(out[2]?.toolResults?.[0]).toMatchObject({ tool_use_id: 'tool-1', isError: false });
    expect(out[3]).toBe(messages[3]);
    expect(out[4]).toBe(messages[4]);
  });

  it('preserves short dropped-result messages and their handles', () => {
    const messages = transcript();
    messages[1]!.toolUses[0]!.text = 'y'.repeat(100);
    messages[2]!.toolResults![0]!.text = 'y'.repeat(100);
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 }),
      decideCall(calls[1]!, { keepCall: 0.9, keepResult: 0.9 }, { keepThreshold: 0.5 }),
    ];
    const out = toSessionMessages(messages, applyDecisions(messages, decisions, calls, 300));
    expect(out[1]).toBe(messages[1]);
    expect(out[2]).toBe(messages[2]);
  });
});

describe('compactSession', () => {
  it('runs the library over the engine fetch and reports the outcome', async () => {
    const bodies: string[] = [];
    const config = {
      ...resolveHookConfig({ backend: 'remote', preserveRecentMessages: 1 }),
      apiKey: 'k',
      model: 'jev-x',
    };
    const { result: output, messages } = await compactSession(
      transcript(),
      config,
      jevFetch((name) => (name === 'call_t2' || name === 'result_t2' ? 0.9 : 0.1), bodies),
    );
    expect(bodies).toHaveLength(1);
    expect(JSON.parse(bodies[0]!).model).toBe('jev-x');
    expect(output.decisions.map((d) => d.action)).toEqual(['drop_call', 'keep']);
    expect(messages.map((m) => m.handle)).toEqual(['h-0', 'h-tool-2', 'r-tool-2', 'h-5', 'h-6']);
    expect(summarize(output)).toMatch(/^\d+% reduction; 1 kept, 1 call_dropped; state ~\d+ tokens \(full\) in 1 request\(s\)$/);
    expect(decisionLog(output)).toBe('t1:Read:drop_call/call=0.10/result=0.10 t2:Bash:keep/call=0.90/result=0.90');
    expect(decisionLogLines(output)).toEqual([`decisions: ${decisionLog(output)}`]);
  });

  it('splits a long decision log into ui.log lines under the host limit', async () => {
    const config = { ...resolveHookConfig({ backend: 'remote', preserveRecentMessages: 1 }), apiKey: 'k' };
    const { result: output } = await compactSession(transcript(), config, jevFetch(() => 0.1));
    const lines = decisionLogLines(output, 60);
    expect(lines).toEqual([
      'decisions (1/2): t1:Read:drop_call/call=0.10/result=0.10',
      'decisions (2/2): t2:Bash:drop_call/call=0.10/result=0.10',
    ]);
    expect(lines.every((line) => line.length <= 60)).toBe(true);
    expect(decisionLogLines({ ...output, decisions: [] })).toEqual(['decisions: (none)']);
  });

  it('throws on a missing key and on failed requests so the hook falls back', async () => {
    const config = resolveHookConfig({ backend: 'remote', preserveRecentMessages: 1 });
    await expect(compactSession(transcript(), config, jevFetch(() => 0))).rejects.toThrow(/TYPESAFE_API_KEY/);
    await expect(
      compactSession(transcript(), { ...config, apiKey: 'k' }, async () => ({ status: 500, ok: false, text: 'x' })),
    ).rejects.toThrow(/500/);
  });
});

describe('compactSession with the local backend', () => {
  it('asks only the local server, needs no key and keeps untouched messages as engine objects', async () => {
    const urls: string[] = [];
    const config = resolveHookConfig({ preserveRecentMessages: 1 });
    expect(config.apiKey).toBeUndefined();
    const { result: output, messages } = await compactSession(
      transcript(),
      config,
      lmStudioFetch((prompt) => (/tool call t2/i.test(prompt.split('[Question]')[1]!) ? 0.9 : 0.1), urls),
    );
    expect(urls).toHaveLength(4);
    expect(new Set(urls)).toEqual(new Set(['http://127.0.0.1:1234/v1/chat/completions']));
    expect(output.decisions.map((d) => d.action)).toEqual(['drop_call', 'keep']);
    expect(messages.map((m) => m.handle)).toEqual(['h-0', 'h-tool-2', 'r-tool-2', 'h-5', 'h-6']);
  });

  it('fails instead of reaching anywhere else when the local server fails', async () => {
    const urls: string[] = [];
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k' };
    await expect(
      compactSession(transcript(), config, async (url) => {
        urls.push(url);
        return { status: 503, ok: false, text: 'no model loaded' };
      }),
    ).rejects.toThrow(/local classifier request failed \(503\)/);
    expect(urls.every((url) => url.startsWith('http://127.0.0.1:1234/'))).toBe(true);
  });

  it('rejects an unknown backend', async () => {
    const config = resolveHookConfig({ backend: 'cloud', preserveRecentMessages: 1 });
    await expect(compactSession(transcript(), config, jevFetch(() => 0))).rejects.toThrow(/unknown backend "cloud"/);
  });
});

type Handler = (...args: unknown[]) => Promise<unknown>;

const CLASSIFY_TOOL = 'mcp__fast-jev-compaction__classify';

function hookEngine(
  options: Record<string, unknown>,
  fetchFn: (url: string, init?: { body?: string }) => Promise<unknown>,
  initialPercent = 0,
) {
  let usagePercent = initialPercent;
  const handlers: Record<string, Handler> = {};
  const matchers: Record<string, unknown> = {};
  register(((event: string, matcherOrHandler: unknown, handler?: Handler) => {
    handlers[event] = handler ?? (matcherOrHandler as Handler);
    if (handler) matchers[event] = matcherOrHandler;
  }) as never, options as never);
  const logs: string[] = [];
  const registered: Array<{ name: string; description: string; inputSchema: unknown }> = [];
  let compactions = 0;
  const $ = {
    ui: { log: (text: string) => logs.push(text), toast: () => {} },
    env: { get: async () => undefined },
    settings: { read: async () => ({}) },
    http: { fetch: fetchFn },
    tool: {
      register: async (spec: { name: string; description: string; inputSchema: unknown }) => {
        registered.push(spec);
        return { tool: `mcp__fast-jev-compaction__${spec.name}` };
      },
    },
    session: {
      usage: async () => ({ context: { percent: usagePercent } }),
      messages: async () => sessionMessages,
      compact: async () => {
        compactions++;
      },
    },
  };
  let sessionMessages: SessionMessage[] = transcript();
  const passthrough = { value: 'next' };
  return {
    logs,
    setPercent: (value: number) => {
      usagePercent = value;
    },
    setMessages: (value: SessionMessage[]) => {
      sessionMessages = value;
    },
    until: async (predicate: () => boolean) => {
      for (let i = 0; i < 500 && !predicate(); i++) await new Promise((resolve) => setTimeout(resolve, 2));
      expect(predicate()).toBe(true);
    },
    matchers,
    registered,
    compactions: () => compactions,
    start: () => handlers['session.start']!($, { cwd: '/', surface: null, isInteractive: false }, async (e: unknown) => e),
    compact: async (messages: SessionMessage[], extra: Record<string, unknown> = {}) => {
      const passed: Array<{ messages: SessionMessage[] } & Record<string, unknown>> = [];
      const out = await handlers['session.compact']!(
        $,
        { trigger: 'manual', messages, ...extra },
        async (e: { messages: SessionMessage[] }) => {
          passed.push(e);
          return { messages: [message('user', 'summary')] };
        },
      );
      return { out, passed };
    },
    call: (input: Record<string, unknown>) =>
      handlers['tool.call']!($, { tool: CLASSIFY_TOOL, tool_use_id: 'x', ...input }, async () => passthrough),
    passthrough,
    turn: (extra: Record<string, unknown> = {}) =>
      handlers['turn.complete']!(
        $,
        { answer: '', durationMs: 1, isAborted: false, turnId: 't', reason: 'answer', ...extra },
        async (e: unknown) => e,
      ),
  };
}

describe('session.compact hook', () => {
  it('hands the pre-compacted transcript to the built-in summary when the reduction is too small', async () => {
    const input = transcript();
    input[1]!.toolUses[0]!.text = 'x'.repeat(2000);
    input[2]!.toolResults![0]!.text = 'x'.repeat(2000);
    const engine = hookEngine(
      { preserveRecentMessages: 1, minReductionRatio: 0.99 },
      lmStudioFetch((prompt) => (/full output of tool call t1/i.test(prompt) ? 0.1 : 0.9)),
    );
    const { passed } = await engine.compact(input);
    expect(passed).toHaveLength(1);
    const handed = passed[0]!.messages;
    expect(handed).toHaveLength(input.length);
    expect(handed[2]!.handle).toBeUndefined();
    expect(handed[2]!.toolResults![0]!.text).toMatch(/fast-jev-compaction truncated 1700 chars/);
    expect(handed[0]).toBe(input[0]);
    expect(engine.logs.at(-1)).toMatch(/^built-in summary over 7\/7 pre-compacted messages \(below 99% minimum/);
  });

  it('replaces the history itself when the reduction is large enough', async () => {
    const input = transcript();
    input[2]!.toolResults![0]!.text = 'x'.repeat(4000);
    const engine = hookEngine({ preserveRecentMessages: 1 }, lmStudioFetch(() => 0.1));
    const { out, passed } = await engine.compact(input);
    expect(passed).toHaveLength(0);
    expect((out as { messages: SessionMessage[] }).messages.map((m) => m.handle)).toEqual(['h-0', 'h-5', 'h-6']);
  });

  it('hands the partially pre-compacted transcript to the summary when the classifier fails midway', async () => {
    const input = transcript();
    input[2]!.toolResults![0]!.text = 'x'.repeat(4000);
    let served = 0;
    const ok = lmStudioFetch(() => 0.1);
    const engine = hookEngine({ preserveRecentMessages: 1, localConcurrency: 1 }, async (url, init) =>
      ++served <= 2 ? ok(url, init) : { status: 500, ok: false, text: 'model crashed' },
    );
    const { passed } = await engine.compact(input);
    expect(passed).toHaveLength(1);
    expect(passed[0]!.messages.map((m) => m.handle)).toEqual(['h-0', 'h-tool-2', 'r-tool-2', 'h-5', 'h-6']);
    expect(engine.logs.at(-1)).toMatch(
      /^built-in summary over 5\/7 partially pre-compacted messages \(local classifier request failed \(500\): model crashed \(1\/2 calls decided\)/,
    );
  });

  it('hands the original transcript to the summary when the classifier is unreachable', async () => {
    const input = transcript();
    const engine = hookEngine({ preserveRecentMessages: 1 }, async () => {
      throw new Error('connect ECONNREFUSED 127.0.0.1:1234');
    });
    const { passed } = await engine.compact(input);
    expect(passed[0]!.messages.every((m, i) => m === input[i])).toBe(true);
    expect(engine.logs.at(-1)).toMatch(/\(0\/2 calls decided\)/);
  });
});

describe('subagents', () => {
  it('leaves a subagent compaction to the built-in summary without asking the classifier', async () => {
    const urls: string[] = [];
    const engine = hookEngine({ preserveRecentMessages: 1 }, lmStudioFetch(() => 0.1, urls));
    const input = transcript();
    const { passed } = await engine.compact(input, { agentId: 'agent-7' });
    expect(urls).toHaveLength(0);
    expect(passed).toHaveLength(1);
    expect(passed[0]!.messages).toBe(input);
    expect(passed[0]!.agentId).toBe('agent-7');
    expect(engine.logs).toEqual(['subagent agent-7 compaction left to the built-in summary']);
  });

  it('at 50% classifies in the background, then compacts on a later main-loop turn', async () => {
    const urls: string[] = [];
    const engine = hookEngine({ preserveRecentMessages: 1 }, lmStudioFetch(() => 0.1, urls), 55);
    await engine.turn({ agentId: 'agent-7' });
    expect(engine.logs).toEqual([]);
    await engine.turn();
    expect(engine.compactions()).toBe(0);
    expect(engine.logs[0]).toBe('background classification started over 7 messages');
    await engine.until(() => engine.logs.some((l) => l.startsWith('background classification done')));
    expect(engine.logs.at(-1)).toMatch(/^background classification done: 2\/2 calls in \d+s$/);
    expect(urls).toHaveLength(4);
    await engine.turn({ agentId: 'agent-7' });
    expect(engine.compactions()).toBe(0);
    await engine.turn();
    expect(engine.compactions()).toBe(1);
  });

  it('does nothing below 50% and drops stale background answers', async () => {
    const engine = hookEngine({ preserveRecentMessages: 1 }, lmStudioFetch(() => 0.1), 55);
    await engine.turn();
    await engine.until(() => engine.logs.some((l) => l.startsWith('background classification done')));
    engine.setPercent(20);
    await engine.turn();
    engine.setPercent(55);
    await engine.turn();
    expect(engine.compactions()).toBe(0);
    expect(engine.logs.filter((l) => l.startsWith('background classification started'))).toHaveLength(2);
  });

  it('compacts right away at 50% with the remote backend', async () => {
    const engine = hookEngine({ backend: 'remote' }, jevFetch(() => 0.1), 55);
    await engine.turn();
    expect(engine.compactions()).toBe(1);
  });
});

function classifyFetch(p: number[], urls: string[] = []) {
  return async (url: string, init?: { body?: string }) => {
    urls.push(url);
    const prompt = (JSON.parse(init?.body ?? '{}') as { messages: Array<{ content: string }> }).messages[0]!.content;
    expect(prompt.endsWith('Answer:')).toBe(true);
    const top = p.map((value, index) => ({ token: ` ${'ABCDEFGHIJ'[index]}`, logprob: Math.log(value) }));
    return {
      status: 200,
      ok: true,
      text: JSON.stringify({ choices: [{ logprobs: { content: [{ top_logprobs: top }] } }], usage: { prompt_tokens: 42 } }),
    };
  };
}

const goodCall = {
  state: 'diff: +return a + b\ntests: 12 passed',
  question: 'Is this implementation acceptable?',
  options: ['acceptable', 'needs changes'],
};

describe('classify tool', () => {
  it('registers classify at session start with its description and schema', async () => {
    const engine = hookEngine({}, classifyFetch([0.5, 0.5]));
    await engine.start();
    expect(engine.registered).toHaveLength(1);
    const [spec] = engine.registered;
    expect(spec!.name).toBe('classify');
    expect(spec!.description).toMatch(/cannot read files/);
    expect(spec!.description).toMatch(/threshold/);
    expect(spec!.description).toMatch(/never the only security check/);
    expect(spec!.inputSchema).toMatchObject({
      type: 'object',
      required: ['state', 'question', 'options'],
      properties: { options: { type: 'array', minItems: 2, maxItems: 10, uniqueItems: true } },
    });
    expect(engine.logs).toEqual([`classify tool registered as ${CLASSIFY_TOOL} (model=jev-style-qwen3.5-2b-decision-mlx)`]);
    expect(engine.matchers['tool.call']).toEqual({ tool: /__classify$/ });
  });

  it('answers with probabilities over the options that sum to 1, only from the local server', async () => {
    const urls: string[] = [];
    const engine = hookEngine({ backend: 'remote', apiKey: 'k' }, classifyFetch([0.2, 0.6, 0.2], urls));
    await engine.start();
    const out = (await engine.call({ ...goodCall, options: ['3', '1', 'unsure'] })) as { result: string };
    expect(Object.keys(out)).toEqual(['result']);
    const probabilityText = out.result.slice(out.result.indexOf('"probabilities"'));
    expect(probabilityText.indexOf('"3"')).toBeLessThan(probabilityText.indexOf('"1"'));
    const parsed = JSON.parse(out.result) as { choice: string; probabilities: Record<string, number> };
    expect(parsed.choice).toBe('1');
    expect(parsed.probabilities['1']).toBeCloseTo(0.6, 10);
    expect(Object.values(parsed.probabilities).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(urls).toEqual(['http://127.0.0.1:1234/v1/chat/completions']);
    expect(engine.logs.at(-1)).toMatch(/^classify: options=3 prompt_tokens=42 ms=\d+ choice="1" p=0\.600$/);
    expect(engine.logs.join('\n')).not.toMatch(/diff|acceptable\?/);
  });

  it('rejects bad input as a tool error without contacting the model', async () => {
    const urls: string[] = [];
    const engine = hookEngine({}, classifyFetch([0.5, 0.5], urls));
    await engine.start();
    const cases: Array<[Record<string, unknown>, RegExp]> = [
      [{ ...goodCall, options: ['only'] }, /2 to 10 strings/],
      [{ ...goodCall, options: Array.from({ length: 11 }, (_, i) => `o${i}`) }, /2 to 10 strings/],
      [{ ...goodCall, options: ['a', 'a'] }, /unique/],
      [{ ...goodCall, options: ['a', ' '] }, /non-empty string/],
      [{ ...goodCall, options: 'a,b' }, /2 to 10 strings/],
      [{ question: goodCall.question, options: goodCall.options }, /state must be/],
      [{ state: goodCall.state, options: goodCall.options }, /question must be/],
      [{ state: goodCall.state, question: goodCall.question }, /2 to 10 strings/],
    ];
    for (const [input, reason] of cases) {
      const out = (await engine.call(input)) as { deny: string };
      expect(out.deny).toMatch(/^classify failed: /);
      expect(out.deny).toMatch(reason);
    }
    expect(urls).toHaveLength(0);
  });

  it('returns every classifier failure as a tool error', async () => {
    const failures: Array<[Record<string, unknown>, (url: string) => Promise<unknown>, RegExp]> = [
      [{}, async () => { throw new Error('connect ECONNREFUSED 127.0.0.1:1234'); }, /unreachable .*ECONNREFUSED/],
      [{}, async () => ({ status: 500, ok: false, text: 'boom' }), /request failed \(500\)/],
      [{}, async () => ({ status: 200, ok: true, text: '{"choices":[{"logprobs":null}]}' }), /no logprobs/],
      [{}, async () => ({ status: 200, ok: true, text: 'not json' }), /malformed JSON/],
      [{ localContextTokens: 10 }, classifyFetch([0.5, 0.5]), /context overflow/],
    ];
    for (const [options, fetchFn, reason] of failures) {
      const engine = hookEngine(options, fetchFn);
      await engine.start();
      const out = (await engine.call(goodCall)) as { deny: string };
      expect(out.deny).toMatch(reason);
      expect(engine.logs.at(-1)).toMatch(/^classify: failed options=2 ms=\d+ /);
    }
  });

  it('can be turned off with classifyTool: false', async () => {
    const urls: string[] = [];
    const engine = hookEngine({ classifyTool: false }, classifyFetch([0.5, 0.5], urls));
    await engine.start();
    expect(engine.registered).toHaveLength(0);
    expect(await engine.call(goodCall)).toBe(engine.passthrough);
    expect(urls).toHaveLength(0);
  });

  it("passes other tools named classify to the next hook", async () => {
    const engine = hookEngine({}, classifyFetch([0.5, 0.5]));
    await engine.start();
    const out = await engine.call({ ...goodCall, tool: 'mcp__other__classify' });
    expect(out).toBe(engine.passthrough);
  });
});

describe('global local queue', () => {
  it('never has more than localConcurrency requests in flight across compactions and classify calls', async () => {
    let inFlight = 0;
    let peak = 0;
    let served = 0;
    const answer = lmStudioFetch(() => 0.9);
    const slow = async (url: string, init?: { body?: string }) => {
      inFlight++;
      peak = Math.max(peak, inFlight);
      await new Promise((resolve) => setTimeout(resolve, 3));
      inFlight--;
      served++;
      return answer(url, init);
    };
    const engine = hookEngine({ preserveRecentMessages: 1, localConcurrency: 2 }, slow);
    await engine.start();
    await Promise.all([
      engine.compact(transcript()),
      engine.compact(transcript()),
      engine.call(goodCall),
      engine.call(goodCall),
      engine.call(goodCall),
    ]);
    expect(served).toBe(4 + 4 + 3);
    expect(peak).toBe(2);
  });
});

describe('staying inside the 10 s hook budget', () => {
  it('compacts from the background answers without asking the model again', async () => {
    const urls: string[] = [];
    const engine = hookEngine({ preserveRecentMessages: 1 }, lmStudioFetch(() => 0.1, urls), 55);
    const input = transcript();
    input[2]!.toolResults![0]!.text = 'x'.repeat(4000);
    engine.setMessages(input);
    await engine.turn();
    await engine.until(() => engine.logs.some((l) => l.startsWith('background classification done')));
    const asked = urls.length;
    const grown = [...input, message('assistant', 'more work', { handle: 'h-7' })];
    const { out, passed } = await engine.compact(grown);
    expect(urls).toHaveLength(asked);
    expect(passed).toHaveLength(0);
    expect(engine.logs).toContainEqual(expect.stringMatching(/^compacting with background answers from \d+s ago$/));
    expect((out as { messages: SessionMessage[] }).messages.map((m) => m.handle)).toEqual(['h-0', 'h-5', 'h-6', 'h-7']);
  });

  it('without background answers, stops at localDeadlineMs and keeps what it did not reach', async () => {
    const answer = lmStudioFetch(() => 0.1);
    const slow = async (url: string, init?: { body?: string }) => {
      await new Promise((resolve) => setTimeout(resolve, 40));
      return answer(url, init);
    };
    const engine = hookEngine({ preserveRecentMessages: 1, localConcurrency: 1, localDeadlineMs: 100 }, slow);
    const input = transcript();
    input[2]!.toolResults![0]!.text = 'x'.repeat(4000);
    const started = Date.now();
    const { out, passed } = await engine.compact(input);
    expect(Date.now() - started).toBeLessThan(400);
    expect(engine.logs).toContainEqual(
      expect.stringMatching(/^compacting with answers from now \(time budget reached: 1\/2 calls decided, the rest kept\)$/),
    );
    expect(passed).toHaveLength(0);
    expect((out as { messages: SessionMessage[] }).messages.map((m) => m.handle)).toEqual([
      'h-0',
      'h-tool-2',
      'r-tool-2',
      'h-5',
      'h-6',
    ]);
  });

  it('answers a classify call that outlasts localDeadlineMs with a tool error', async () => {
    const engine = hookEngine({ localDeadlineMs: 30 }, async (url, init) => {
      await new Promise((resolve) => setTimeout(resolve, 200));
      return classifyFetch([0.5, 0.5])(url, init);
    });
    await engine.start();
    const started = Date.now();
    const out = (await engine.call(goodCall)) as { deny: string };
    expect(Date.now() - started).toBeLessThan(150);
    expect(out.deny).toMatch(/classify failed: local classifier decision did not finish within its time budget/);
  });
});

