// free-model-aggregator-01 entry: one Node server exposing OpenAI and Anthropic client wires over a best-free-model routing layer with retry-aware provider failover
//
// Startup: probe every provider catalog, register with the harness, then answer the client's 20 manifest requests.
// goes to the best-rated free model that satisfies its requirements and still has budget; when an upstream
// answers 402 / 429 / 5xx, fail over to the next candidate. The reply comes back in whatever wire the
// client spoke (JSON or SSE), keeping the provider's stop reason, usage and echo text intact.
//
// Deterministic throughout: no wall clock, no randomness; every decision is a pure function of the brief and
// providers' answers. GET /v1/models replies in the wire named by the auth header. `main.ts --list`
// emits the exact catalog the register body carries, byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type RatingRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  sseEvent,
  type ServedModel,
} from './lib/wire.ts';
import {
  geminiAnswerOf,
  geminiStreamToAnthropic,
  geminiStreamToOpenAI,
  geminiToAnthropic,
  geminiToOpenAI,
  geminiSseBodyToAnswer,
  openAIToGemini,
} from './lib/gemini.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const UPSTREAM_TIMEOUT_MS = 25000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models the responding providers serve, which is what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Route {
  probe: Probe;
  model: ProbedModel;
  row: RatingRow | null;
}

/** Candidate routes for a request's requirements, best-rated first, ties resolved by provider order. */
function candidatesFor(requires: Requirement): Route[] {
  const out: Route[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, row });
    }
  }
  out.sort((a, b) => {
    const ra = a.row?.rating_milli ?? -1;
    const rb = b.row?.rating_milli ?? -1;
    if (ra !== rb) return rb - ra;
    const ia = probes.indexOf(a.probe);
    const ib = probes.indexOf(b.probe);
    if (ia !== ib) return ia - ib;
    return a.model.id < b.model.id ? -1 : 1;
  });
  return out;
}

/**
 * Choose one candidate: the best-rated one whose provider still has budget left in this trial. A
 * provider that refused with 402 or an outage is skipped for the rest of the trial; a 429 is retried
 * only after the others (its budget may simply be spent in order).
 */
const refusedHard = new Set<string>(); // provider ids refused with 402 / outage / repeated failures
const callCount = new Map<string, number>(); // provider id -> successful + failed counted calls this trial

function pickRoute(requires: Requirement): Route | null {
  const candidates = candidatesFor(requires);
  if (candidates.length === 0) return null;
  // Favor candidates whose providers appear to still have budget, then the rest in rating order.
  const fresh = candidates.filter((c) => !refusedHard.has(c.probe.provider.id));
  return fresh[0] ?? candidates[0];
}

function noteCall(providerId: string): void {
  callCount.set(providerId, (callCount.get(providerId) ?? 0) + 1);
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamJson {
  ok: boolean;
  status: number;
  retryAfterMs: number | null;
  text: string;
}

/** A single upstream call in the provider's own wire, returning the raw status and body text. */
async function callUpstream(route: Route, clientBody: Json, clientApi: 'openai' | 'anthropic'): Promise<UpstreamJson> {
  const provider = route.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  const stream = clientBody.stream === true;
  let url: string;
  let body: string;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    const out: Json = clientApi === 'openai' ? { ...clientBody, model: route.model.id } : anthropicToOpenAI(clientBody, route.model.id);
    out.model = route.model.id;
    out.stream = stream;
    if (stream) out.stream_options = { include_usage: true };
    body = JSON.stringify(out);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    const anthropic: Json = clientApi === 'anthropic' ? { ...clientBody, model: route.model.id } : anthropicToOpenAI(clientBody, route.model.id);
    // When the client is OpenAI but the provider is Anthropic, translate the OpenAI body over.
    const converted = clientApi === 'anthropic' ? anthropic : openAIToAnthropicRequest(clientBody, route.model.id);
    converted.stream = stream;
    body = JSON.stringify(converted);
    headers['anthropic-version'] = '2023-06-01';
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(route.model.id)}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
    if (stream) url += '?alt=sse';
    const converted = clientApi === 'openai' ? openAIToGemini(clientBody) : anthropicToGeminiRequest(clientBody, route.model.id);
    body = JSON.stringify(converted);
  }
  if (stream) headers.accept = 'text/event-stream';
  try {
    const res = await fetch(url, { method: 'POST', headers, body, signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
    const text = await res.text();
    let retryAfterMs: number | null = null;
    const ra = res.headers.get('retry-after');
    if (ra !== null) {
      const secs = Number(ra);
      if (Number.isFinite(secs) && secs >= 0) retryAfterMs = secs * 1000;
    }
    noteCall(provider.id);
    if (res.status === 402) refusedHard.add(provider.id);
    return { ok: res.ok, status: res.status, retryAfterMs, text };
  } catch (err) {
    noteCall(provider.id);
    return { ok: false, status: 0, retryAfterMs: null, text: (err as Error).message };
  }
}

function stopReasonOfFinish(finish: unknown): string {
  if (finish === 'length') return 'length';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_calls';
  if (finish === 'content_filter') return 'content_filter';
  return 'stop';
}

/** Parse an OpenAI Chat Completions answer (JSON or SSE) into its text, finish reason, usage and tool calls. */
function openAIAnswerOf(text: string, stream: boolean): { text: string; finish: string | null; input: number | null; output: number | null; toolCalls: Json[] } | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(text);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const message = isRecord(choice.message) ? choice.message : {};
    const content = typeof message.content === 'string' ? message.content : '';
    const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls.filter(isRecord) : [];
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text: content,
      finish: typeof choice.finish_reason === 'string' ? choice.finish_reason : null,
      input: typeof usage.prompt_tokens === 'number' ? usage.prompt_tokens : null,
      output: typeof usage.completion_tokens === 'number' ? usage.completion_tokens : null,
      toolCalls,
    };
  }
  // SSE: collect the chunks.
  let out = '';
  let finish: string | null = null;
  let input: number | null = null;
  let output: number | null = null;
  const toolCalls = new Map<number, Json>();
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    for (const line of frame.split('\n')) {
      if (!line.startsWith('data:')) continue;
      const payload = line.slice(5).trim();
      if (payload === '[DONE]') continue;
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        input = typeof chunk.usage.prompt_tokens === 'number' ? chunk.usage.prompt_tokens : input;
        output = typeof chunk.usage.completion_tokens === 'number' ? chunk.usage.completion_tokens : output;
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string') out += delta.content;
      if (typeof choice.finish_reason === 'string' && choice.finish_reason !== '') finish = choice.finish_reason;
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const index = typeof call.index === 'number' ? call.index : toolCalls.size;
        const existing = toolCalls.get(index);
        const fn = isRecord(call.function) ? call.function : {};
        if (existing === undefined) {
          toolCalls.set(index, {
            id: typeof call.id === 'string' ? call.id : `call_${index}`,
            type: 'function',
            function: { name: typeof fn.name === 'string' ? fn.name : '', arguments: typeof fn.arguments === 'string' ? fn.arguments : '' },
          });
        } else {
          const efn = isRecord(existing.function) ? existing.function : {};
          if (typeof fn.arguments === 'string') efn.arguments = (typeof efn.arguments === 'string' ? efn.arguments : '') + fn.arguments;
        }
      }
    }
  }
  return { text: out, finish, input, output, toolCalls: [...toolCalls.values()] };
}

/** An Anthropic Messages answer's text, stop reason and usage, from JSON or SSE text. */
function anthropicAnswerOf(text: string, stream: boolean): { text: string; finish: string | null; input: number | null; output: number | null; toolCalls: Json[] } | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(text);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    const content = Array.isArray(value.content) ? value.content : [];
    let out = '';
    const toolCalls: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') out += block.text;
      if (block.type === 'tool_use') toolCalls.push(block);
    }
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text: out,
      finish: typeof value.stop_reason === 'string' ? value.stop_reason : null,
      input: typeof usage.input_tokens === 'number' ? usage.input_tokens : null,
      output: typeof usage.output_tokens === 'number' ? usage.output_tokens : null,
      toolCalls,
    };
  }
  let out = '';
  let finish: string | null = null;
  let input: number | null = null;
  let output: number | null = null;
  let curTool: Json | null = null;
  const toolCalls: Json[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    let event: string | null = null;
    const datas: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) datas.push(line.slice(5).replace(/^ /, ''));
    }
    if (datas.length === 0) continue;
    let data: unknown;
    try {
      data = JSON.parse(datas.join('\n'));
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = typeof data.message.usage.input_tokens === 'number' ? data.message.usage.input_tokens : input;
    }
    if (event === 'content_block_start' && isRecord(data.content_block) && data.content_block.type === 'tool_use') {
      curTool = { ...data.content_block, input: isRecord(data.content_block.input) ? data.content_block.input : {} };
    }
    if (event === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') out += data.delta.text;
      if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string' && curTool !== null) {
        curTool.partialJson = (typeof curTool.partialJson === 'string' ? curTool.partialJson : '') + data.delta.partial_json;
      }
    }
    if (event === 'content_block_stop' && curTool !== null) {
      const partial = typeof curTool.partialJson === 'string' && curTool.partialJson.trim() !== '' ? curTool.partialJson : '{}';
      let inputObj: unknown = {};
      try {
        inputObj = JSON.parse(partial);
      } catch {
        inputObj = {};
      }
      toolCalls.push({ ...curTool, input: isRecord(inputObj) ? inputObj : {} });
      curTool = null;
    }
    if (event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') finish = data.delta.stop_reason;
      if (isRecord(data.usage) && typeof data.usage.output_tokens === 'number') output = data.usage.output_tokens;
    }
  }
  return { text: out, finish, input, output, toolCalls };
}

/** One client request answered with failover: try candidates until one succeeds. */
async function serve(req: IncomingMessage, res: ServerResponse, api: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, api, 400, api === 'anthropic' ? 'invalid_request_error' : 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  const tries: Route[] = [];
  const first = pickRoute(requires);
  if (first === null) {
    return sendError(res, api, api === 'anthropic' ? 529 : 503, api === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  tries.push(first);
  // Build the full failover list lazily: after each failure, the next untried candidate.
  const all = candidatesFor(requires).filter((c) => c !== first);

  let attempt = 0;
  let lastFailure: { status: number; text: string } | null = null;
  for (;;) {
    const route = attempt === 0 ? first : nextCandidate(all, tries);
    if (route === null) break;
    tries.push(route);
    const upstream = await callUpstream(route, body, api);
    if (upstream.ok) {
      const delivered = deliver(res, api, route, body, stream, upstream.text);
      if (delivered) return;
      lastFailure = { status: upstream.status, text: upstream.text.slice(0, 300) };
    } else {
      lastFailure = { status: upstream.status, text: upstream.text.slice(0, 300) };
      // A 429 on a provider we have not exhausted may mean concurrency or budget; try the next one.
      if (upstream.status === 402) refusedHard.add(route.probe.provider.id);
      if (upstream.retryAfterMs !== null && upstream.retryAfterMs > 0) {
        // Respect the provider's retry-after: move on to another provider now (never a wall-clock wait
        // tied to output, a bounded sleep is a function of the provider's answer alone).
        await sleep(Math.min(upstream.retryAfterMs, 2000));
      }
    }
    attempt += 1;
    if (attempt > 24) break;
  }
  const status = lastFailure?.status === 0 ? 502 : lastFailure !== null ? 502 : 503;
  sendError(res, api, status, api === 'anthropic' ? 'api_error' : 'upstream_error', lastFailure !== null ? `every provider that meets this request refused: HTTP ${lastFailure.status}` : 'no free model meets this request');
}

/** The next candidate not yet tried, preferring providers without a hard refusal. */
function nextCandidate(all: Route[], tries: Route[]): Route | null {
  const triedProviders = new Set(tries.map((t) => `${t.probe.provider.id}/${t.model.id}`));
  const fresh = all.filter((c) => !triedProviders.has(`${c.probe.provider.id}/${c.model.id}`) && !refusedHard.has(c.probe.provider.id));
  if (fresh.length > 0) return fresh[0];
  const rest = all.filter((c) => !triedProviders.has(`${c.probe.provider.id}/${c.model.id}`));
  return rest[0] ?? null;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Deliver one successful upstream answer in the client's wire; false when the answer is unusable. */
function deliver(res: ServerResponse, api: 'openai' | 'anthropic', route: Route, body: Json, stream: boolean, upstreamText: string): boolean {
  const provider = route.probe.provider;
  try {
    if (provider.wire === 'openai') {
      const answer = openAIAnswerOf(upstreamText, stream);
      if (answer === null || answer.input === null) return false;
      return emitAnswer(res, api, stream, answer.text, answer.finish, answer.input, answer.output, answer.toolCalls, route.model.id);
    }
    if (provider.wire === 'anthropic') {
      const answer = anthropicAnswerOf(upstreamText, stream);
      if (answer === null || answer.input === null) return false;
      return emitAnswer(res, api, stream, answer.text, answer.finish, answer.input, answer.output, answer.toolCalls, route.model.id);
    }
    // Gemini provider.
    const answer = stream ? geminiSseBodyToAnswer(upstreamText) : (geminiAnswerOf(safeParse(upstreamText)) ?? null);
    if (answer === null || answer.input === null) return false;
    const finish = answer.finish === null ? 'stop' : answer.finish;
    return emitAnswer(res, api, stream, answer.text, finish, answer.input, answer.output, answer.toolCalls, route.model.id);
  } catch {
    return false;
  }
}

function safeParse(text: string): Json {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : {};
  } catch {
    return {};
  }
}

const FINISH_TO_OPENAI: Record<string, string> = { stop: 'stop', end_turn: 'stop', STOP: 'stop', length: 'length', max_tokens: 'length', MAX_TOKENS: 'length', tool_calls: 'tool_calls', tool_use: 'tool_calls', function_call: 'tool_calls', content_filter: 'content_filter', refusal: 'content_filter', SAFETY: 'content_filter' };

/** Emit the answer to the client in its wire, JSON or SSE, with usage and stop reason mapped. */
function emitAnswer(
  res: ServerResponse,
  api: 'openai' | 'anthropic',
  stream: boolean,
  text: string,
  finish: string | null,
  input: number,
  output: number | null,
  toolCalls: Json[],
  model: string,
): boolean {
  const outFinish = finish === null ? 'stop' : finish;
  if (api === 'openai') {
    const openaiFinish = FINISH_TO_OPENAI[outFinish] ?? 'stop';
    const usage = { prompt_tokens: input, completion_tokens: output ?? Math.max(1, Math.ceil(text.length / 4)), total_tokens: input + (output ?? Math.max(1, Math.ceil(text.length / 4))) };
    const message: Json = { role: 'assistant', content: text };
    if (toolCalls.length > 0) message.tool_calls = toolCalls;
    if (!stream) {
      sendJson(res, 200, {
        id: 'chatcmpl-sim',
        object: 'chat.completion',
        created: 1767225600,
        model,
        choices: [{ index: 0, message, finish_reason: openaiFinish, logprobs: null }],
        usage,
      });
      return true;
    }
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    const chunk = (delta: Json, fr: string | null, u: Json | null) => ({
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: 1767225600,
      model,
      choices: [{ index: 0, delta, finish_reason: fr }],
      ...(u !== null ? { usage: u } : {}),
    });
    const pieces = splitText(text);
    let out = `data: ${JSON.stringify(chunk({ role: 'assistant', content: '' }, null, null))}\n\n`;
    for (const piece of pieces) out += `data: ${JSON.stringify(chunk({ content: piece }, null, null))}\n\n`;
    out += `data: ${JSON.stringify(chunk({}, openaiFinish, null))}\n\n`;
    out += `data: ${JSON.stringify(chunk({}, null, usage))}\n\n`;
    out += 'data: [DONE]\n\n';
    res.end(out);
    return true;
  }
  // The client is Anthropic.
  const stopReason = FINISH_TO_OPENAI[outFinish] === 'stop' ? 'end_turn' : FINISH_TO_OPENAI[outFinish] === 'length' ? 'max_tokens' : FINISH_TO_OPENAI[outFinish] === 'tool_calls' ? 'tool_use' : 'refusal';
  const outTokens = output ?? Math.max(1, Math.ceil(text.length / 4));
  if (!stream) {
    const content: Json[] = [];
    if (text !== '') content.push({ type: 'text', text });
    for (const call of toolCalls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      let args: Json = {};
      if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
        try {
          const parsed: unknown = JSON.parse(call.function.arguments);
          if (isRecord(parsed)) args = parsed;
        } catch {
          args = {};
        }
      } else if (isRecord(call.input)) args = call.input;
      content.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : 'toolu_unknown', name: typeof call.function.name === 'string' ? call.function.name : '', input: args });
    }
    if (content.length === 0) content.push({ type: 'text', text: '' });
    sendJson(res, 200, {
      id: 'msg_sim',
      type: 'message',
      role: 'assistant',
      model,
      content,
      stop_reason: stopReason,
      stop_sequence: null,
      usage: { input_tokens: input, output_tokens: outTokens },
    });
    return true;
  }
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  const pieces = splitText(text);
  let out = sseEvent('message_start', { type: 'message_start', message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: input, output_tokens: 0 } } });
  out += sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  for (const piece of pieces) out += sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } });
  out += sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  out += sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outTokens } });
  out += sseEvent('message_stop', { type: 'message_stop' });
  res.end(out);
  return true;
}

const SSE_DELTA_CHARS = 13;

function splitText(text: string): string[] {
  if (text === '') return [''];
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += SSE_DELTA_CHARS) pieces.push(text.slice(at, at + SSE_DELTA_CHARS));
  return pieces;
}

/** An OpenAI Chat Completions request as an Anthropic Messages request (for anthropic-wire providers). */
function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const messagesIn = Array.isArray(body.messages) ? body.messages : [];
  let system = '';
  for (const raw of messagesIn) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      system = typeof raw.content === 'string' ? raw.content : system;
      continue;
    }
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const blocks: Json[] = [];
      const text = content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
      if (text !== '') blocks.push({ type: 'text', text });
      for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : 'toolu_unknown', name: call.function.name, input: safeParseObj(call.function.arguments) });
      }
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    const blocks: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
      else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
        const url = block.image_url.url;
        const data = /^data:([^,]*);base64,(.*)$/s.exec(url);
        if (data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] || 'image/png', data: data[2] } });
        else blocks.push({ type: 'image', source: { type: 'url', url } });
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    if (typeof raw.tool_call_id === 'string') {
      // An OpenAI tool result message: a user message with a tool_result block.
      const result = typeof raw.content === 'string' ? raw.content : JSON.stringify(raw.content ?? '');
      messages[messages.length - 1] = { role: 'user', content: [{ type: 'tool_result', tool_use_id: raw.tool_call_id, content: result }] };
    }
  }
  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  else out.max_tokens = 512;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop.filter((s) => typeof s === 'string');
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function)) out.tool_choice = { type: 'tool', name: choice.function.name };
  return out;
}

function safeParseObj(text: unknown): Json {
  if (typeof text === 'string' && text.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(text);
      if (isRecord(parsed)) return parsed;
    } catch {
      // fall through
    }
  }
  return {};
}

/** An Anthropic Messages request as a Gemini generateContent body (for gemini-wire providers). */
function anthropicToGeminiRequest(body: Json, model: string): Json {
  const openai = anthropicToOpenAI(body, model);
  return openAIToGemini(openai);
}

async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  await serve(req, res, 'openai');
}

async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  await serve(req, res, 'anthropic');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: retry on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
