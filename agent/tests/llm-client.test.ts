import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

describe('llm-client', () => {
  it('module imports without error', async () => {
    const mod = await import('../src/llm-client.js');
    assert.equal(typeof mod.chatCompletion, 'function');
  });
});
