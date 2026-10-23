// Builds one Promptfoo test per row of the held-out test set.
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');

module.exports = async function () {
  const systemPrompt = fs.readFileSync(
    path.join(ROOT, 'registry', 'adapters', 'config_A', 'system_prompt.txt'), 'utf-8');
  const lines = fs.readFileSync(path.join(ROOT, 'training', 'colab', 'test.jsonl'), 'utf-8')
    .split('\n').filter((l) => l.trim());

  return lines.map((line, i) => {
    const row = JSON.parse(line);
    const gold = typeof row.output === 'string' ? JSON.parse(row.output) : row.output;
    return {
      description: `row ${i + 1}`,
      vars: {
        index: i + 1,
        system_prompt: systemPrompt,
        transcript: row.input,
        gold_codes: (gold.suggested_codes || []).map((c) => String(c.code)),
      },
    };
  });
};
