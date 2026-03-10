import type { ToolResult } from './types.js';

const RESET = '\x1b[0m';
const DIM = '\x1b[2m';
const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const CYAN = '\x1b[36m';
const YELLOW = '\x1b[33m';
const BOLD = '\x1b[1m';

/** Display a tool call being executed (interactive mode) */
export function printToolCall(name: string, args: Record<string, unknown>): void {
  const argsStr = JSON.stringify(args, null, 0);
  const truncated = argsStr.length > 120 ? argsStr.slice(0, 117) + '...' : argsStr;
  process.stderr.write(`${CYAN}\u25b6${RESET} ${BOLD}${name}${RESET}${DIM}(${truncated})${RESET}\n`);
}

/** Display tool result (interactive mode) */
export function printToolResult(name: string, result: ToolResult, elapsed: number): void {
  const text = result.content.map(c => c.text).join('\n');
  const lines = text.split('\n').slice(0, 5);
  const prefix = result.isError ? `${RED}\u2717${RESET}` : `${GREEN}\u2713${RESET}`;
  const elapsedStr = `${DIM}(${(elapsed / 1000).toFixed(1)}s)${RESET}`;

  for (const line of lines) {
    process.stderr.write(`  ${line}\n`);
  }
  if (text.split('\n').length > 5) {
    process.stderr.write(`  ${DIM}... (${text.split('\n').length - 5} more lines)${RESET}\n`);
  }
  process.stderr.write(`  ${prefix} completed ${elapsedStr}\n\n`);
}

/** Display agent text message in a box (interactive mode) */
export function printAgentMessage(text: string): void {
  const lines = text.split('\n');
  const maxLen = Math.min(Math.max(...lines.map(l => l.length)), 60);
  const border = '\u2500'.repeat(maxLen + 2);

  process.stderr.write(`\n${CYAN}\u250c\u2500 Agent ${border.slice(8)}\u2510${RESET}\n`);
  for (const line of lines) {
    process.stderr.write(`${CYAN}\u2502${RESET} ${line.padEnd(maxLen)} ${CYAN}\u2502${RESET}\n`);
  }
  process.stderr.write(`${CYAN}\u2514${border}\u2518${RESET}\n`);
}

/** Display tool call in compact mode (--yes) */
export function printCompact(name: string, summary: string): void {
  const shortName = name.replace('mloop_', '');
  process.stderr.write(`${DIM}[${shortName}]${RESET} ${summary}\n`);
}

/** Display warning */
export function printWarning(msg: string): void {
  process.stderr.write(`${YELLOW}\u26a0 ${msg}${RESET}\n`);
}

/** Display error */
export function printError(msg: string): void {
  process.stderr.write(`${RED}\u2717 ${msg}${RESET}\n`);
}

/** Read user input from stdin */
export function readUserInput(prompt: string = '> '): Promise<string> {
  return new Promise((resolve) => {
    process.stderr.write(prompt);
    process.stdin.setEncoding('utf8');
    process.stdin.once('data', (chunk: string) => {
      resolve(chunk.trim());
    });
  });
}

/** Summarize tool result for compact mode */
export function summarizeResult(name: string, result: ToolResult): string {
  const text = result.content.map(c => c.text).join(' ');
  const firstLine = text.split('\n').find(l => l.trim().length > 0) || '';
  return firstLine.length > 80 ? firstLine.slice(0, 77) + '...' : firstLine;
}
