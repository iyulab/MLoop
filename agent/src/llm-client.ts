import type { ChatMessage, ChatCompletionResponse, ToolDefinition } from './types.js';

export interface LLMClientConfig {
  endpoint: string;  // e.g. http://localhost:11434/v1
  model: string;     // e.g. llama3.1
}

/**
 * Send a chat completion request to an OpenAI-compatible API.
 * Uses native fetch() — no SDK dependency.
 */
export async function chatCompletion(
  config: LLMClientConfig,
  messages: ChatMessage[],
  tools?: ToolDefinition[],
): Promise<ChatCompletionResponse> {
  const url = `${config.endpoint.replace(/\/+$/, '')}/chat/completions`;

  const body: Record<string, unknown> = {
    model: config.model,
    messages,
  };

  if (tools && tools.length > 0) {
    body['tools'] = tools;
    body['tool_choice'] = 'auto';
  }

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const text = await response.text().catch(() => '');
    throw new Error(`LLM API error ${response.status}: ${text}`);
  }

  return response.json() as Promise<ChatCompletionResponse>;
}
