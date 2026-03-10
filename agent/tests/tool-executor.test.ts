import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { getToolDefinitions } from '../src/tool-executor.js';

describe('tool-executor', () => {
  describe('getToolDefinitions', () => {
    it('returns array of OpenAI-compatible tool definitions', () => {
      const tools = getToolDefinitions();
      assert.ok(tools.length > 0, 'Should have at least one tool');

      for (const tool of tools) {
        assert.equal(tool.type, 'function');
        assert.ok(tool.function.name, 'Tool must have a name');
        assert.ok(tool.function.description, 'Tool must have a description');
        assert.ok(tool.function.parameters, 'Tool must have parameters');
        assert.equal(typeof tool.function.parameters, 'object');
      }
    });

    it('includes core tools: info, init, train, predict, list', () => {
      const tools = getToolDefinitions();
      const names = tools.map(t => t.function.name);
      assert.ok(names.includes('mloop_info'));
      assert.ok(names.includes('mloop_init'));
      assert.ok(names.includes('mloop_train'));
      assert.ok(names.includes('mloop_predict'));
      assert.ok(names.includes('mloop_list'));
    });

    it('excludes composite workflow tools', () => {
      const tools = getToolDefinitions();
      const names = tools.map(t => t.function.name);
      assert.ok(!names.includes('mloop_quick_start'));
      assert.ok(!names.includes('mloop_auto_train'));
      assert.ok(!names.includes('mloop_project_overview'));
    });

    it('all tools have projectPath in schema properties', () => {
      const tools = getToolDefinitions();
      for (const tool of tools) {
        const props = (tool.function.parameters as Record<string, unknown>)['properties'] as Record<string, unknown>;
        assert.ok(props?.['projectPath'], `${tool.function.name} should have projectPath`);
      }
    });
  });
});
