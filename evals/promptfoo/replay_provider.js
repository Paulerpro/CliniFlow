// Replay provider: returns the reply already saved by evals/run_eval.py for this
// test row instead of calling a server, so assertions can be built and rerun for
// free (no GPU). Config: { resultsFile: "<path relative to evals/>" }.
const fs = require('fs');
const path = require('path');

class ReplayProvider {
  constructor(options) {
    this.providerId = options.id || 'replay';
    const file = path.resolve(__dirname, '..', options.config.resultsFile);
    this.results = JSON.parse(fs.readFileSync(file, 'utf-8')).results;
  }

  id() {
    return this.providerId;
  }

  async callApi(prompt, context) {
    const row = this.results.find((r) => r.index === Number(context.vars.index));
    if (!row || row.output == null) {
      return { error: `no saved output for row ${context.vars.index} (request failed in the original run)` };
    }
    return { output: row.output, latencyMs: row.latency_s * 1000 };
  }
}

module.exports = ReplayProvider;
