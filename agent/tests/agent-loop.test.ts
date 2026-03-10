import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

describe('agent-loop', () => {
  it('module imports without error', async () => {
    const mod = await import('../src/agent-loop.js');
    assert.equal(typeof mod.runAgent, 'function');
  });

  describe('DONE pattern', () => {
    const DONE_PATTERN = /\bdone\b/i;

    it('matches "DONE"', () => assert.ok(DONE_PATTERN.test('DONE')));
    it('matches "Done!"', () => assert.ok(DONE_PATTERN.test('Done!')));
    it('matches "DONE: summary"', () => assert.ok(DONE_PATTERN.test('DONE: Model built')));
    it('matches "done."', () => assert.ok(DONE_PATTERN.test('I am done.')));
    it('does not match "undone"', () => assert.ok(!DONE_PATTERN.test('This is undone')));
    it('does not match "done" inside a word', () => assert.ok(!DONE_PATTERN.test('abandoned')));
  });
});
