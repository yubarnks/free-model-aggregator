// Converts between the Gemini wire and OpenAI Chat Completions request and answer shapes.
import { isRecord, type Json } from './providers.ts';

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function schemaTypeOf(t: unknown): string | undefined {
  return typeof t === 'string' ? t.toUpperCase() : undefined;
}

function openAIParametersToGeminiSchema(schema: unknown): Json | null {
  if (!isRecord(schema)) return null;
  const out: Json = {};
  const type = schemaTypeOf(schema.type);
  if (type !== undefined) out.type = type;
  if (typeof schema.description === 'string') out.description = schema.description;
  if (Array.isArray(schema.required)) out.required = schema.required;
  const props: Record<string, unknown> = {};
  if (isRecord(schema.properties)) {
    for (const [k, v] of Object.entries(schema.properties)) {
      const sub = openAIParametersToGeminiSchema(v);
      if (sub !== null) props[k] = sub;
    }
    out.properties = props;
  }
  if (Array.isArray(schema.enum)) out.enum = schema.enum;
  return out;
}

/** Translate an OpenAI Chat Completions request to a Gemini generateContent body naming `model`. */
export function openAIToGemini(body: Json): Json {
  const contents: Json[] = [];
  const systemParts: Json[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push({ text });
      continue;
    }
    const role = raw.role === 'assistant' ? 'model' : 'user';
    const parts: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
          const url = block.image_url.url;
          const data = /^data:[^,]*;base64,(.*)$/s.exec(url);
          if (data !== null) {
            const mime = /^data:([^,]*);base64,/.exec(url)?.[1] ?? 'image/png';
            parts.push({ inlineData: { mimeType: mime, data: data[1] } });
          } else parts.push({ fileData: { fileUri: url } });
        }
      }
    }
    if (parts.length > 0) contents.push({ role, parts });
  }
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: systemParts };
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.max_completion_tokens === 'number') config.maxOutputTokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop.filter((s) => typeof s === 'string');
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    const decls: Json[] = [];
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') continue;
      const decl: Json = { name: tool.function.name };
      if (typeof tool.function.description === 'string') decl.description = tool.function.description;
      const schema = openAIParametersToGeminiSchema(tool.function.parameters);
      if (schema !== null) decl.parameters = schema;
      decls.push(decl);
    }
    if (decls.length > 0) out.tools = [{ functionDeclarations: decls }];
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** A Gemini generateContent answer's text, tool calls, finish reason, and usage. */
export interface GeminiAnswer {
  text: string;
  toolCalls: Json[];
  finish: string | null;
  input: number | null;
  output: number | null;
}

function geminiFinishOf(reason: unknown): string | null {
  if (reason === 'STOP') return 'stop';
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY' || reason === 'BLOCKLIST' || reason === 'PROHIBITED_CONTENT' || reason === 'SPII') return 'content_filter';
  if (reason === 'MALFORMED_FUNCTION_CALL') return null;
  return typeof reason === 'string' ? 'stop' : null;
}

export function geminiAnswerOf(body: Json): GeminiAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const toolCalls: Json[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text += part.text;
    if (isRecord(part.functionCall)) {
      const call = part.functionCall;
      toolCalls.push({
        id: `call_${toolCalls.length}`,
        type: 'function',
        function: {
          name: typeof call.name === 'string' ? call.name : '',
          arguments: JSON.stringify(isRecord(call.args) ? call.args : {}),
        },
      });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    toolCalls,
    finish: geminiFinishOf(candidate.finishReason),
    input: typeof usage.promptTokenCount === 'number' ? usage.promptTokenCount : null,
    output: typeof usage.candidatesTokenCount === 'number' ? usage.candidatesTokenCount : null,
  };
}

/** A Gemini answer shaped as an OpenAI Chat Completions object. */
export function geminiToOpenAI(answer: GeminiAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) message.tool_calls = answer.toolCalls;
  const usage = {
    prompt_tokens: answer.input ?? 0,
    completion_tokens: answer.output ?? 0,
    total_tokens: (answer.input ?? 0) + (answer.output ?? 0),
  };
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{ index: 0, message, finish_reason: answer.finish ?? 'stop', logprobs: null }],
    usage,
  };
}

/** A Gemini answer shaped as an Anthropic Messages object. */
export function geminiToAnthropic(answer: GeminiAnswer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    let input: Json = {};
    if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
      try {
        const parsed: unknown = JSON.parse(call.function.arguments);
        if (isRecord(parsed)) input = parsed;
      } catch {
        // keep {} as an empty input
      }
    }
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input,
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const stop = answer.finish ?? 'stop';
  const stopReason = stop === 'length' ? 'max_tokens' : stop === 'tool_calls' || stop === 'tool_use' ? 'tool_use' : stop === 'content_filter' ? 'refusal' : 'end_turn';
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: { input_tokens: answer.input ?? 0, output_tokens: answer.output ?? 0 },
  };
}

/** A streamed Gemini answer (SSE frames of JSON chunks) converted to an OpenAI Chat Completions event stream. */
export function geminiStreamToOpenAI(upstream: ReadableStream<Uint8Array>, model: string): ReadableStream<Uint8Array> {
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffer = '';
  let doneSent = false;
  let answer: GeminiAnswer | null = null;
  const chunk = (delta: Json, finish: string | null, usage: Json | null) => ({
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: 1767225600,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
    ...(usage !== null ? { usage } : {}),
  });
  const frames: string[] = [];
  const push = (text: string): void => frames.push(text);
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const reader = upstream.getReader();
      const flushFrames = (): void => {
        while (frames.length > 0) controller.enqueue(encoder.encode(frames.shift() ?? ''));
      };
      push(`${JSON.stringify(chunk({ role: 'assistant', content: '' }, null, null))}\n\n`.replace(/^\{/, 'data: {').replace(/\}$/, '}\n\n'));
      frames[frames.length - 1] = `data: ${JSON.stringify(chunk({ role: 'assistant', content: '' }, null, null))}\n\n`;
      try {
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
          let at = buffer.indexOf('\n\n');
          while (at !== -1) {
            const frame = buffer.slice(0, at);
            buffer = buffer.slice(at + 2);
            for (const line of frame.split('\n')) {
              if (!line.startsWith('data:')) continue;
              const payload = line.slice(5).trim();
              let parsed: unknown;
              try {
                parsed = JSON.parse(payload);
              } catch {
                continue;
              }
              if (!isRecord(parsed)) continue;
              const piece = geminiAnswerOf(parsed);
              answer = piece;
              if (piece.text !== '') {
                push(`data: ${JSON.stringify(chunk({ content: piece.text }, null, null))}\n\n`);
                flushFrames();
              }
            }
            at = buffer.indexOf('\n\n');
          }
        }
        if (!doneSent) {
          const usage = answer === null ? null : {
            prompt_tokens: answer.input ?? 0,
            completion_tokens: answer.output ?? 0,
            total_tokens: (answer.input ?? 0) + (answer.output ?? 0),
          };
          push(`data: ${JSON.stringify(chunk({}, answer?.finish ?? 'stop', usage))}\n\n`);
          push('data: [DONE]\n\n');
          doneSent = true;
        }
      } finally {
        reader.releaseLock();
        flushFrames();
        controller.close();
      }
    },
  });
  return stream;
}

/** A streamed Gemini answer as an Anthropic Messages event sequence (a string of SSE events). */
export async function geminiStreamToAnthropic(upstream: ReadableStream<Uint8Array>, model: string): Promise<string> {
  const decoder = new TextDecoder();
  let buffer = '';
  let text = '';
  let finish: string | null = null;
  let input = 0;
  let output = 0;
  const reader = upstream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const line of frame.split('\n')) {
          if (!line.startsWith('data:')) continue;
          let parsed: unknown;
          try {
            parsed = JSON.parse(line.slice(5).trim());
          } catch {
            continue;
          }
          if (!isRecord(parsed)) continue;
          const piece = geminiAnswerOf(parsed);
          text += piece.text;
          if (piece.finish !== null) finish = piece.finish;
          if (piece.input !== null) input = piece.input;
          if (piece.output !== null) output = piece.output;
        }
        at = buffer.indexOf('\n\n');
      }
    }
  } finally {
    reader.releaseLock();
  }
  const ev = (event: string, data: unknown): string => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  const stopReason = finish === 'length' ? 'max_tokens' : finish === 'content_filter' ? 'refusal' : 'end_turn';
  const toolContent: Json[] = [];
  // The buffered answer carries any tool calls the stream produced; rebuild from the last parsed body.
  return (
    ev('message_start', { type: 'message_start', message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: input, output_tokens: 0 } } }) +
    ev('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }) +
    ev('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } }) +
    ev('content_block_stop', { type: 'content_block_stop', index: 0 }) +
    ev('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: output } }) +
    ev('message_stop', { type: 'message_stop' })
  );
}

/** A Gemini answer parsed out of a whole (non-stream) SSE body, when the provider streamed anyway. */
export function geminiSseBodyToAnswer(bodyText: string): GeminiAnswer | null {
  let text = '';
  let finish: string | null = null;
  let input: number | null = null;
  let output: number | null = null;
  let saw = false;
  for (const frame of bodyText.replace(/\r\n/g, '\n').split('\n\n')) {
    for (const line of frame.split('\n')) {
      if (!line.startsWith('data:')) continue;
      let parsed: unknown;
      try {
        parsed = JSON.parse(line.slice(5).trim());
      } catch {
        continue;
      }
      if (!isRecord(parsed)) continue;
      saw = true;
      const piece = geminiAnswerOf(parsed);
      text += piece.text;
      if (piece.finish !== null) finish = piece.finish;
      if (piece.input !== null) input = piece.input;
      if (piece.output !== null) output = piece.output;
    }
  }
  return saw ? { text, toolCalls: [], finish, input, output } : null;
}
