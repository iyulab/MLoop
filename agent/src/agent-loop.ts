import type { AgentConfig, ChatMessage } from './types.js';
import { chatCompletion } from './llm-client.js';
import { getToolDefinitions, executeTool } from './tool-executor.js';
import {
  printToolCall, printToolResult, printAgentMessage,
  printCompact, printWarning, printError,
  readUserInput, summarizeResult,
} from './ui.js';

const DONE_PATTERN = /\bdone\b/i;

function buildSystemPrompt(config: AgentConfig): string {
  let prompt = `You are an ML engineer assistant. You have access to MLoop tools for building ML.NET models. Your task is to build the best possible model from the provided data file.

Workflow:
1. Use mloop_info to understand the dataset
2. Use mloop_init to create a project (if needed)
3. Use mloop_train to train a model
4. Use mloop_evaluate or mloop_list to check results
5. If results are poor, try different parameters
6. Use mloop_predict for final predictions
7. Say "DONE" with a summary when finished

IMPORTANT: Always include projectPath in tool calls: ${config.projectPath}
Data file (absolute path): ${config.dataFile}`;

  if (config.label) {
    prompt += `\nLabel column: ${config.label}`;
  }
  if (config.task) {
    prompt += `\nTask type: ${config.task}`;
  }

  return prompt;
}

function buildInitialUserMessage(config: AgentConfig): string {
  const parts = [`Build an ML model from ${config.dataFile}.`];
  if (config.label) parts.push(`Target column: ${config.label}.`);
  if (config.task) parts.push(`Task: ${config.task}.`);
  return parts.join(' ');
}

/**
 * Run the agent loop.
 * Returns exit code: 0 = success (DONE), 1 = max-turns exhausted or error
 */
export async function runAgent(config: AgentConfig): Promise<number> {
  const tools = getToolDefinitions();
  const messages: ChatMessage[] = [
    { role: 'system', content: buildSystemPrompt(config) },
    { role: 'user', content: buildInitialUserMessage(config) },
  ];

  const llmConfig = { endpoint: config.endpoint, model: config.model };

  for (let turn = 0; turn < config.maxTurns; turn++) {
    let response;
    try {
      response = await chatCompletion(llmConfig, messages, tools);
    } catch (err) {
      printError(`LLM API call failed: ${err instanceof Error ? err.message : String(err)}`);
      return 1;
    }

    const choice = response.choices[0];
    if (!choice) {
      printError('LLM returned empty response');
      return 1;
    }

    const assistantMsg = choice.message;

    // Append assistant message to history
    messages.push(assistantMsg);

    // Handle tool calls
    if (assistantMsg.tool_calls && assistantMsg.tool_calls.length > 0) {
      for (const toolCall of assistantMsg.tool_calls) {
        let args: Record<string, unknown>;
        try {
          args = JSON.parse(toolCall.function.arguments);
        } catch {
          args = {};
        }

        if (config.yes) {
          // Compact mode
          const result = await executeTool(toolCall.function.name, args, config.projectPath);
          printCompact(toolCall.function.name, summarizeResult(toolCall.function.name, result));
          messages.push({
            role: 'tool',
            tool_call_id: toolCall.id,
            content: result.content.map(c => c.text).join('\n'),
          });
        } else {
          // Interactive mode
          printToolCall(toolCall.function.name, args);
          const start = Date.now();
          const result = await executeTool(toolCall.function.name, args, config.projectPath);
          const elapsed = Date.now() - start;
          printToolResult(toolCall.function.name, result, elapsed);
          messages.push({
            role: 'tool',
            tool_call_id: toolCall.id,
            content: result.content.map(c => c.text).join('\n'),
          });
        }
      }
      continue;
    }

    // Handle text response (no tool calls)
    const textContent = assistantMsg.content || '';

    // Check for completion
    if (DONE_PATTERN.test(textContent)) {
      if (config.yes) {
        printCompact('done', textContent.split('\n')[0] || 'Complete');
      } else {
        printAgentMessage(textContent);
      }
      return 0;
    }

    // Not done — interact with user or auto-continue
    if (config.yes) {
      // Auto-continue in --yes mode
      messages.push({ role: 'user', content: 'continue' });
    } else {
      // Interactive: show message, get user input
      printAgentMessage(textContent);
      const input = await readUserInput();
      if (input.toLowerCase() === 'quit' || input.toLowerCase() === 'exit') {
        return 0;
      }
      messages.push({ role: 'user', content: input });
    }
  }

  // Max turns exhausted
  printWarning(`Max turns (${config.maxTurns}) reached. Exiting.`);
  return 1;
}
