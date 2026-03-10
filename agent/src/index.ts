#!/usr/bin/env node

import path from 'node:path';
import fs from 'node:fs';
import type { AgentConfig } from './types.js';
import { runAgent } from './agent-loop.js';

function printUsage(): void {
  console.error(`Usage: mloop-agent run <data-file> [options]

Options:
  --endpoint <url>    LLM API endpoint (default: http://localhost:11434/v1)
  --model <name>      Model name (default: llama3.1)
  --label <column>    Target column name
  --task <type>       ML task: binary-classification, multiclass-classification, regression
  --project <dir>     Project directory (default: ./mloop-project)
  --yes               Skip confirmation prompts (fully autonomous)
  --max-turns <n>     Maximum agent loop iterations (default: 30)
  --verbose           Show full LLM responses
  --help              Show this help`);
}

function parseArgs(argv: string[]): AgentConfig | 'help' | null {
  const args = argv.slice(2); // Skip node and script path

  if (args.includes('--help') || args.includes('-h')) {
    printUsage();
    return 'help';
  }

  if (args.length === 0) {
    printUsage();
    return null;
  }

  // Expect: run <data-file> [options]
  if (args[0] !== 'run') {
    console.error(`Unknown command: ${args[0]}. Use "mloop-agent run <data-file>".`);
    return null;
  }

  if (!args[1] || args[1].startsWith('--')) {
    console.error('Error: data file path is required.');
    printUsage();
    return null;
  }

  const dataFile = path.resolve(args[1]);

  // Parse flags
  const config: AgentConfig = {
    dataFile,
    endpoint: 'http://localhost:11434/v1',
    model: 'llama3.1',
    projectPath: path.resolve('./mloop-project'),
    yes: false,
    maxTurns: 30,
    verbose: false,
  };

  for (let i = 2; i < args.length; i++) {
    const flag = args[i];
    switch (flag) {
      case '--endpoint':
        config.endpoint = args[++i] || config.endpoint;
        break;
      case '--model':
        config.model = args[++i] || config.model;
        break;
      case '--label':
        config.label = args[++i];
        break;
      case '--task':
        config.task = args[++i];
        break;
      case '--project':
        config.projectPath = path.resolve(args[++i] || './mloop-project');
        break;
      case '--yes':
      case '-y':
        config.yes = true;
        break;
      case '--max-turns':
        config.maxTurns = parseInt(args[++i] || '30', 10);
        break;
      case '--verbose':
        config.verbose = true;
        break;
      default:
        console.error(`Unknown flag: ${flag}`);
        printUsage();
        return null;
    }
  }

  return config;
}

async function main(): Promise<void> {
  const config = parseArgs(process.argv);
  if (config === 'help') {
    process.exit(0);
  }
  if (!config) {
    process.exit(1);
  }

  // Validate data file exists
  if (!fs.existsSync(config.dataFile)) {
    console.error(`Error: Data file not found: ${config.dataFile}`);
    process.exit(1);
  }

  const exitCode = await runAgent(config);
  process.exit(exitCode);
}

main();
