import { describe, expect, it } from 'vitest';
import {
  buildLocalRequest,
  compact,
  decisionPrompt,
  LOCAL_BASE_URL,
  LOCAL_MODEL,
  LocalJevAsker,
  parseLocalResponse,
  createScheduler,
  classifyCalls,
  compactWithAnswers,
  PartialAnswersError,
  PartialCompactionError,
  type LocalFetch,
  type Message,
} from '../src/index.js';

type Top = Array<{ token: string; logprob: number }>;

function completion(top: Top, promptTokens = 100): string {
  return JSON.stringify({
    choices: [
      {
        message: { role: 'assistant', content: top[0]?.token.trim() ?? '' },
        logprobs: { content: [{ token: top[0]?.token, logprob: top[0]?.logprob, top_logprobs: top }] },
        finish_reason: 'length',
      },
    ],
    usage: { prompt_tokens: promptTokens, completion_tokens: 1 },
  });
}

/** LM Studio's shape for two options with P(A) = p, plus some off-option noise. */
function topFor(p: number): Top {
  return [
    { token: ' A', logprob: Math.log(p * 0.9) },
    { token: ' B', logprob: Math.log((1 - p) * 0.9) },
    { token: ' C', logprob: Math.log(0.05) },
    { token: '<|endoftext|>', logprob: Math.log(0.05) },
  ];
}

type Call = { url: string; body: { model: string; messages: Array<{ content: string }> } & Record<string, unknown> };

function lmStudio(answer: (prompt: string) => number, calls: Call[] = []): LocalFetch {
  return async (url, init) => {
    const body = JSON.parse(init.body) as Call['body'];
    calls.push({ url, body });
    return { status: 200, ok: true, text: completion(topFor(answer(body.messages[0]!.content))) };
  };
}

describe('decision prompt and request', () => {
  it('builds the exact Jev-style prompt ending in Answer:', () => {
    expect(decisionPrompt({ a: 1 }, 'Is it true?', ['yes', 'no'])).toBe(
      'You are a decision function. Read the state, then answer the question by choosing exactly one option.\n\n' +
        '[State]\n{"a":1}\n\n[Question]\nIs it true?\n\n[Options]\nA. yes\nB. no\n\nAnswer:',
    );
    expect(decisionPrompt('plain', 'q', ['x', 'y', 'z'])).toContain('[State]\nplain\n\n');
  });

  it('asks LM Studio for one greedy token with its top log-probs', () => {
    const request = buildLocalRequest({}, 'PROMPT');
    expect(request.url).toBe(`${LOCAL_BASE_URL}/chat/completions`);
    expect(request.url).toBe('http://127.0.0.1:1234/v1/chat/completions');
    expect(request.headers).toEqual({ 'content-type': 'application/json' });
    expect(JSON.parse(request.body)).toEqual({
      model: LOCAL_MODEL,
      messages: [{ role: 'user', content: 'PROMPT' }],
      temperature: 0,
      max_tokens: 2,
      logprobs: true,
      top_logprobs: 10,
    });
    const custom = buildLocalRequest({ baseUrl: 'http://localhost:9999/v1/', model: 'm' }, 'P');
    expect(custom.url).toBe('http://localhost:9999/v1/chat/completions');
    expect(JSON.parse(custom.body).model).toBe('m');
  });
});

describe('parseLocalResponse', () => {
  it('renormalises the option letters over the declared options only', () => {
    const { probabilities, promptTokens } = parseLocalResponse(200, true, completion(topFor(0.83)), 2);
    expect(probabilities[0]).toBeCloseTo(0.83, 10);
    expect(probabilities[1]).toBeCloseTo(0.17, 10);
    expect(promptTokens).toBe(100);
  });

  it('matches the real LM Studio MLX response', () => {
    const top = [
      { token: ' A', logprob: -0.375 },
      { token: ' B', logprob: -1.625 },
      { token: ' C', logprob: -3 },
      { token: ' D', logprob: -3.8125 },
      { token: '<|endoftext|>', logprob: -6.0625 },
    ];
    const [a, b] = parseLocalResponse(200, true, completion(top), 2).probabilities;
    expect(a).toBeCloseTo(1 / (1 + Math.exp(-1.25)), 10);
    expect(a! + b!).toBeCloseTo(1, 12);
  });

  it('gives a declared letter outside the candidates negligible mass', () => {
    const top = [
      { token: ' A', logprob: -0.01 },
      { token: ' C', logprob: -6 },
    ];
    const [a, b] = parseLocalResponse(200, true, completion(top), 2).probabilities;
    expect(a).toBeGreaterThan(0.999);
    expect(b).toBeGreaterThan(0);
  });

  it('adds up the mass of two spellings of one letter', () => {
    const top = [
      { token: ' A', logprob: Math.log(0.3) },
      { token: 'A', logprob: Math.log(0.3) },
      { token: ' B', logprob: Math.log(0.4) },
    ];
    expect(parseLocalResponse(200, true, completion(top), 2).probabilities[0]).toBeCloseTo(0.6, 10);
  });

  it('throws on every malformed or failed response', () => {
    const parse = (text: string, ok = true, status = 200) => () => parseLocalResponse(status, ok, text, 2);
    expect(parse('boom', false, 500)).toThrow(/request failed \(500\)/);
    expect(
      parse(
        '{"error":"The number of tokens to keep from the initial prompt is greater than the context length."}',
        false,
        400,
      ),
    ).toThrow(/context overflow \(400\)/);
    expect(parse('not json')).toThrow(/malformed JSON/);
    expect(parse('{}')).toThrow(/no choices/);
    expect(parse('{"choices":[]}')).toThrow(/no choices/);
    expect(parse('{"choices":[{"logprobs":null}]}')).toThrow(/no logprobs/);
    expect(parse('{"choices":[{"logprobs":{"content":[]}}]}')).toThrow(/no candidate tokens/);
    expect(parse('{"choices":[{"logprobs":{"content":[{"top_logprobs":[{"token":1}]}]}}]}')).toThrow(
      /malformed logprobs/,
    );
    expect(parse(completion([{ token: 'The', logprob: -0.1 }]))).toThrow(/none of the options A\/B/);
  });
});

describe('LocalJevAsker', () => {
  it('answers each noul question with P(A) from its own prompt, without any key', async () => {
    const calls: Call[] = [];
    const asker = new LocalJevAsker({ fetch: lmStudio((p) => (p.includes('keep me') ? 0.9 : 0.2), calls) });
    const response = await asker.ask(
      { history: [] },
      {
        keep: { type: 'noul', instructions: 'keep me' },
        drop: { type: 'noul', instructions: 'drop me', criteria: { true: 'it stays', false: 'it goes' } },
      },
    );
    expect(response.model).toBe(LOCAL_MODEL);
    expect((response.answers.keep as { noul: number }).noul).toBeCloseTo(0.9, 10);
    expect((response.answers.drop as { noul: number }).noul).toBeCloseTo(0.2, 10);
    expect(response.usage).toEqual({ input_tokens: 200, output_tokens: 2 });
    expect(calls).toHaveLength(2);
    expect(calls.every((c) => c.url === 'http://127.0.0.1:1234/v1/chat/completions')).toBe(true);
    expect(calls[0]!.body.messages[0]!.content).toMatch(/\[Question\]\nkeep me\n\n\[Options\]\nA\. yes\nB\. no\n\nAnswer:$/);
    expect(calls[1]!.body.messages[0]!.content).toMatch(/A\. it stays\nB\. it goes\n\nAnswer:$/);
  });

  it('refuses question types it cannot answer', async () => {
    const asker = new LocalJevAsker({ fetch: lmStudio(() => 0.5) });
    await expect(
      asker.ask('s', { c: { type: 'choice', instructions: 'pick', criteria: { a: null } } }),
    ).rejects.toThrow(/only noul questions \(c is choice\)/);
  });

  it('refuses a prompt over the context budget before sending it', async () => {
    const calls: Call[] = [];
    const asker = new LocalJevAsker({ fetch: lmStudio(() => 0.5, calls), contextTokens: 50 });
    await expect(asker.ask('word '.repeat(200), { q: { type: 'noul', instructions: 'q' } })).rejects.toThrow(
      /context overflow: prompt ~\d+ tokens exceeds 50/,
    );
    expect(calls).toHaveLength(0);
  });

  it('rejects a non-positive context budget as misconfiguration', () => {
    expect(() => new LocalJevAsker({ fetch: lmStudio(() => 0.5), contextTokens: 0 })).toThrow(
      /contextTokens must be positive \(got 0\)/,
    );
  });

  it('reports an unreachable server', async () => {
    const asker = new LocalJevAsker({
      fetch: async () => {
        throw new Error('connect ECONNREFUSED 127.0.0.1:1234');
      },
    });
    await expect(asker.ask('s', { q: { type: 'noul', instructions: 'q' } })).rejects.toThrow(
      /unreachable at http:\/\/127\.0\.0\.1:1234\/v1\/chat\/completions: connect ECONNREFUSED/,
    );
  });

  it('keeps at most `concurrency` requests in flight across concurrent asks', async () => {
    let inFlight = 0;
    let peak = 0;
    let served = 0;
    const fetch: LocalFetch = async () => {
      inFlight++;
      peak = Math.max(peak, inFlight);
      await new Promise((resolve) => setTimeout(resolve, 2));
      inFlight--;
      served++;
      return { status: 200, ok: true, text: completion(topFor(0.7)) };
    };
    const asker = new LocalJevAsker({ fetch, concurrency: 2 });
    const questions = Object.fromEntries(
      Array.from({ length: 5 }, (_, i) => [`q${i}`, { type: 'noul' as const, instructions: `q${i}` }]),
    );
    await Promise.all([asker.ask('s', questions), asker.ask('s', questions), asker.ask('s', questions)]);
    expect(served).toBe(15);
    expect(peak).toBe(2);
  });

  it('fails with the answers it got before the failure', async () => {
    let served = 0;
    const fetch: LocalFetch = async () =>
      ++served <= 2
        ? { status: 200, ok: true, text: completion(topFor(0.8)) }
        : { status: 500, ok: false, text: 'model crashed' };
    const asker = new LocalJevAsker({ fetch, concurrency: 1 });
    const questions = Object.fromEntries(
      Array.from({ length: 4 }, (_, i) => [`q${i}`, { type: 'noul' as const, instructions: `q${i}` }]),
    );
    const error = await asker.ask('s', questions).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(PartialAnswersError);
    expect((error as PartialAnswersError).message).toMatch(/request failed \(500\)/);
    expect(Object.keys((error as PartialAnswersError).answers).sort()).toEqual(['q0', 'q1']);
    expect(served).toBe(3);
  });

  it('stops sending the rest of a batch after a failure', async () => {
    let served = 0;
    const fetch: LocalFetch = async () => {
      served++;
      return { status: 500, ok: false, text: 'model crashed' };
    };
    const asker = new LocalJevAsker({ fetch, concurrency: 1 });
    const questions = Object.fromEntries(
      Array.from({ length: 4 }, (_, i) => [`q${i}`, { type: 'noul' as const, instructions: `q${i}` }]),
    );
    await expect(asker.ask('s', questions)).rejects.toThrow(/request failed \(500\): model crashed/);
    expect(served).toBe(1);
  });
});

describe('compact with the local classifier', () => {
  const fileA = 'export const a = 1;\n'.repeat(50);
  const transcript: Message[] = [
    { role: 'user', text: 'Fix the failing test.', toolUses: [] },
    { role: 'assistant', text: '', toolUses: [{ tool_use_id: 'u1', tool: 'Read', input: { file_path: 'a.ts' }, text: fileA }] },
    { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'u1', text: fileA }] },
    { role: 'assistant', text: '', toolUses: [{ tool_use_id: 'u2', tool: 'Bash', input: { command: 'npm test' }, text: 'FAIL' }] },
    { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'u2', text: 'FAIL b.test.ts', isError: true }] },
    { role: 'assistant', text: 'Fixing now.', toolUses: [] },
  ];

  it('applies keepThreshold to the renormalised probabilities and leaves kept messages untouched', async () => {
    const calls: Call[] = [];
    const answer = (prompt: string) =>
      prompt.includes('output of tool call t1') ? 0.3 : prompt.includes('Tool call t1') ? 0.6 : 0.95;
    const asker = new LocalJevAsker({ fetch: lmStudio(answer, calls) });
    const result = await compact(transcript, asker, { preserveRecentMessages: 1, keepThreshold: 0.5 });
    expect(result.decisions.map((d) => [d.id, d.action])).toEqual([
      ['t1', 'drop_result'],
      ['t2', 'keep'],
    ]);
    expect(result.decisions[0]!.keepCall).toBeCloseTo(0.6, 10);
    expect(result.decisions[0]!.keepResult).toBeCloseTo(0.3, 10);
    expect(calls).toHaveLength(4);
    for (const index of [0, 3, 4, 5]) expect(result.messages[index]).toBe(transcript[index]);
    expect(result.messages[2]!.toolResults![0]!.text).toMatch(/fast-jev-compaction truncated/);

    const strict = await compact(transcript, new LocalJevAsker({ fetch: lmStudio(answer) }), {
      preserveRecentMessages: 1,
      keepThreshold: 0.97,
    });
    expect(strict.decisions.map((d) => d.action)).toEqual(['drop_call', 'drop_call']);
  });
});

describe('compact when the local classifier fails midway', () => {
  const big = 'export const a = 1;\n'.repeat(50);
  const transcript: Message[] = [
    { role: 'user', text: 'Fix it.', toolUses: [] },
    { role: 'assistant', text: '', toolUses: [{ tool_use_id: 'u1', tool: 'Read', input: { f: 1 }, text: big }] },
    { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'u1', text: big }] },
    { role: 'assistant', text: '', toolUses: [{ tool_use_id: 'u2', tool: 'Read', input: { f: 2 }, text: big }] },
    { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'u2', text: big }] },
    { role: 'assistant', text: 'Done.', toolUses: [] },
  ];

  it('applies the answers that came back, keeps the rest and still throws', async () => {
    let served = 0;
    const fetch: LocalFetch = async () =>
      ++served <= 2
        ? { status: 200, ok: true, text: completion(topFor(0.1)) }
        : { status: 500, ok: false, text: 'model crashed' };
    const error = await compact(transcript, new LocalJevAsker({ fetch, concurrency: 1 }), {
      preserveRecentMessages: 1,
    }).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(PartialCompactionError);
    const { result } = error as PartialCompactionError;
    expect((error as Error).message).toMatch(/request failed \(500\): model crashed \(1\/2 calls decided\)/);
    expect(result.decisions.map((d) => [d.id, d.action])).toEqual([
      ['t1', 'drop_call'],
      ['t2', 'keep'],
    ]);
    expect(result.messages).toEqual([transcript[0], transcript[3], transcript[4], transcript[5]]);
    for (const [index, original] of [[1, 3], [2, 4]] as const) expect(result.messages[index]).toBe(transcript[original]);
  });

  it('keeps everything when no answer came back', async () => {
    const fetch: LocalFetch = async () => {
      throw new Error('connect ECONNREFUSED 127.0.0.1:1234');
    };
    const error = (await compact(transcript, new LocalJevAsker({ fetch }), { preserveRecentMessages: 1 }).catch(
      (e: unknown) => e,
    )) as PartialCompactionError;
    expect(error.message).toMatch(/unreachable .* \(0\/2 calls decided\)/);
    expect(error.result.messages).toEqual(transcript);
    expect(error.result.messages.every((m, i) => m === transcript[i])).toBe(true);
  });
});

describe('LocalJevAsker.decide', () => {
  it('returns the renormalised distribution over 2 to 10 options and the most probable one', async () => {
    const calls: Call[] = [];
    const fetch: LocalFetch = async (url, init) => {
      calls.push({ url, body: JSON.parse(init.body) });
      return {
        status: 200,
        ok: true,
        text: completion([
          { token: ' A', logprob: Math.log(0.1) },
          { token: ' B', logprob: Math.log(0.3) },
          { token: ' C', logprob: Math.log(0.4) },
          { token: ' D', logprob: Math.log(0.2) },
        ]),
      };
    };
    const decision = await new LocalJevAsker({ fetch }).decide('s', 'Which?', ['x', 'y', 'z']);
    expect(decision.choice).toBe('z');
    expect(decision.probabilities.map((p) => p.toFixed(4))).toEqual(['0.1250', '0.3750', '0.5000']);
    expect(decision.probabilities.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(calls[0]!.body.messages[0]!.content).toMatch(/\[Options\]\nA\. x\nB\. y\nC\. z\n\nAnswer:$/);
  });

  it('validates the question and the options', async () => {
    const asker = new LocalJevAsker({ fetch: lmStudio(() => 0.5) });
    await expect(asker.decide('s', ' ', ['a', 'b'])).rejects.toThrow(/question must be/);
    await expect(asker.decide('s', 'q', ['a'])).rejects.toThrow(/2 to 10 strings/);
    await expect(asker.decide('s', 'q', Array.from({ length: 11 }, (_, i) => `${i}`))).rejects.toThrow(/2 to 10/);
    await expect(asker.decide('s', 'q', ['a', ' a'])).rejects.toThrow(/unique/);
    await expect(asker.decide('s', 'q', ['a', ''])).rejects.toThrow(/non-empty/);
  });

  it('shares one scheduler between askers', async () => {
    let inFlight = 0;
    let peak = 0;
    const fetch: LocalFetch = async () => {
      inFlight++;
      peak = Math.max(peak, inFlight);
      await new Promise((resolve) => setTimeout(resolve, 2));
      inFlight--;
      return { status: 200, ok: true, text: completion(topFor(0.7)) };
    };
    const scheduler = createScheduler(2);
    const one = new LocalJevAsker({ fetch, scheduler, concurrency: 8 });
    const two = new LocalJevAsker({ fetch, scheduler, concurrency: 8 });
    const questions = Object.fromEntries(
      Array.from({ length: 4 }, (_, i) => [`q${i}`, { type: 'noul' as const, instructions: `q${i}` }]),
    );
    await Promise.all([one.ask('s', questions), two.ask('s', questions), one.decide('s', 'q', ['a', 'b'])]);
    expect(peak).toBe(2);
  });
});

describe('priority, deadline and abort', () => {
  it('runs queued high-priority work before queued low-priority work', async () => {
    const order: string[] = [];
    const schedule = createScheduler(1);
    const task = (name: string) => async () => {
      order.push(name);
      await new Promise((resolve) => setTimeout(resolve, 2));
    };
    await Promise.all([
      schedule(task('first'), 'low'),
      schedule(task('low-1'), 'low'),
      schedule(task('low-2'), 'low'),
      schedule(task('high'), 'high'),
    ]);
    expect(order).toEqual(['first', 'high', 'low-1', 'low-2']);
  });

  it('answers with what it has at the deadline, flagged as timed out', async () => {
    const fetch: LocalFetch = async () => {
      await new Promise((resolve) => setTimeout(resolve, 30));
      return { status: 200, ok: true, text: completion(topFor(0.7)) };
    };
    const asker = new LocalJevAsker({ fetch, concurrency: 1, deadline: Date.now() + 45 });
    const questions = Object.fromEntries(
      Array.from({ length: 5 }, (_, i) => [`q${i}`, { type: 'noul' as const, instructions: `q${i}` }]),
    );
    const error = (await asker.ask('s', questions).catch((e: unknown) => e)) as PartialAnswersError;
    expect(error).toBeInstanceOf(PartialAnswersError);
    expect(error.timedOut).toBe(true);
    expect(error.message).toMatch(/time budget reached \(1\/5 questions answered\)/);
    expect(Object.keys(error.answers)).toEqual(['q0']);
  });

  it('stops sending once its signal is aborted', async () => {
    let served = 0;
    const controller = new AbortController();
    const fetch: LocalFetch = async () => {
      served++;
      controller.abort();
      return { status: 200, ok: true, text: completion(topFor(0.7)) };
    };
    const asker = new LocalJevAsker({ fetch, concurrency: 1, signal: controller.signal });
    const questions = Object.fromEntries(
      Array.from({ length: 4 }, (_, i) => [`q${i}`, { type: 'noul' as const, instructions: `q${i}` }]),
    );
    const error = (await asker.ask('s', questions).catch((e: unknown) => e)) as PartialAnswersError;
    expect(error.timedOut).toBe(true);
    expect(error.message).toMatch(/stopped \(1\/4 questions answered\)/);
    expect(served).toBe(1);
  });
});

describe('classifyCalls and compactWithAnswers', () => {
  const big = 'export const a = 1;\n'.repeat(50);
  const transcript: Message[] = [
    { role: 'user', text: 'Fix it.', toolUses: [] },
    { role: 'assistant', text: '', toolUses: [{ tool_use_id: 'u1', tool: 'Read', input: { f: 1 }, text: big }] },
    { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'u1', text: big }] },
    { role: 'assistant', text: 'Done.', toolUses: [] },
  ];

  it('gives the same result as compact, and applies earlier answers to a longer transcript', async () => {
    const fetch = lmStudio(() => 0.1);
    const direct = await compact(transcript, new LocalJevAsker({ fetch }), { preserveRecentMessages: 1 });
    const classification = await classifyCalls(transcript, new LocalJevAsker({ fetch }), { preserveRecentMessages: 1 });
    expect([...classification.answers.keys()]).toEqual(['u1']);
    const later = compactWithAnswers(transcript, classification, { preserveRecentMessages: 1 });
    expect(later.messages).toEqual(direct.messages);
    expect(later.decisions).toEqual(direct.decisions);
    const grown: Message[] = [
      ...transcript,
      { role: 'assistant', text: '', toolUses: [{ tool_use_id: 'u2', tool: 'Read', input: { f: 2 }, text: big }] },
      { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'u2', text: big }] },
      { role: 'assistant', text: 'More.', toolUses: [] },
    ];
    const applied = compactWithAnswers(grown, classification, { preserveRecentMessages: 1 });
    expect(applied.decisions.map((d) => [d.id, d.action])).toEqual([
      ['t1', 'drop_call'],
      ['t2', 'keep'],
    ]);
    expect(applied.messages.slice(-3)).toEqual(grown.slice(-3));
  });
});

