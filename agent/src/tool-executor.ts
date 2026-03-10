import path from 'node:path';
import {
  trainTool, train,
  predictTool, predict,
  listTool, list,
  promoteTool, promote,
  infoTool, info,
  statusTool, status,
  compareTool, compare,
  evaluateTool, evaluate,
  initTool, init,
  validateTool, validate,
  prepTool, prep,
  logsTool, logs,
  feedbackTool, feedback,
  sampleTool, sample,
  triggerTool, trigger,
} from '@iyulab/mloop-mcp/tools';
import type { ToolDefinition, ToolResult } from './types.js';

/** Tool descriptor with handler */
interface ToolEntry {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
  handler: (params: Record<string, unknown>) => Promise<ToolResult>;
}

/**
 * All available tools for the agent.
 * Excluded: composite workflow tools (quick_start, auto_train, project_overview)
 * because the agent LLM should decide the workflow steps itself.
 * Also excluded: serve (not needed for model building workflow).
 */
const toolRegistry: ToolEntry[] = [
  { ...infoTool, handler: info as ToolEntry['handler'] },
  { ...initTool, handler: init as ToolEntry['handler'] },
  { ...trainTool, handler: train as ToolEntry['handler'] },
  { ...predictTool, handler: predict as ToolEntry['handler'] },
  { ...listTool, handler: list as ToolEntry['handler'] },
  { ...promoteTool, handler: promote as ToolEntry['handler'] },
  { ...statusTool, handler: status as ToolEntry['handler'] },
  { ...compareTool, handler: compare as ToolEntry['handler'] },
  { ...evaluateTool, handler: evaluate as ToolEntry['handler'] },
  { ...validateTool, handler: validate as ToolEntry['handler'] },
  { ...prepTool, handler: prep as ToolEntry['handler'] },
  { ...logsTool, handler: logs as ToolEntry['handler'] },
  { ...feedbackTool, handler: feedback as ToolEntry['handler'] },
  { ...sampleTool, handler: sample as ToolEntry['handler'] },
  { ...triggerTool, handler: trigger as ToolEntry['handler'] },
];

const handlerMap = new Map<string, ToolEntry['handler']>(
  toolRegistry.map(t => [t.name, t.handler])
);

/** Build OpenAI-compatible tools array for LLM */
export function getToolDefinitions(): ToolDefinition[] {
  return toolRegistry.map(tool => ({
    type: 'function' as const,
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.inputSchema,
    },
  }));
}

/**
 * Execute a tool call with auto-injected projectPath.
 * - Injects projectPath if not provided by LLM
 * - Resolves relative dataFile paths to absolute
 */
export async function executeTool(
  toolName: string,
  rawArgs: Record<string, unknown>,
  defaultProjectPath: string,
): Promise<ToolResult> {
  const handler = handlerMap.get(toolName);
  if (!handler) {
    return {
      content: [{ type: 'text', text: `Unknown tool: ${toolName}` }],
      isError: true,
    };
  }

  const args = { ...rawArgs };

  // Auto-inject projectPath
  if (!args['projectPath']) {
    args['projectPath'] = defaultProjectPath;
  }

  // Resolve relative dataFile to absolute
  if (args['dataFile'] && typeof args['dataFile'] === 'string' && !path.isAbsolute(args['dataFile'])) {
    args['dataFile'] = path.resolve(defaultProjectPath, args['dataFile']);
  }

  return handler(args);
}
