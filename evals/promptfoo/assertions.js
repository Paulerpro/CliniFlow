// Custom assertions. They mirror evals/run_eval.py so Promptfoo and the Python
// scripts agree on what "valid" means.
const SOAP = ['subjective', 'objective', 'assessment', 'plan'];
const CODE_FIELDS = ['code', 'system', 'description', 'confidence'];

function parseLenient(text) {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  const candidates = fenced ? [fenced[1]] : [];
  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if (start !== -1 && end > start) candidates.push(text.slice(start, end + 1));
  for (const c of candidates) {
    try {
      const obj = JSON.parse(c);
      if (obj && typeof obj === 'object' && !Array.isArray(obj)) return obj;
    } catch (e) { /* try next */ }
  }
  return null;
}

function schemaProblems(obj) {
  const problems = [];
  for (const f of SOAP) if (typeof obj[f] !== 'string') problems.push(`${f} not a string`);
  const codes = obj.suggested_codes;
  if (!Array.isArray(codes) || codes.length === 0) {
    problems.push('suggested_codes missing or empty');
  } else if (codes.some((c) => !c || CODE_FIELDS.some((f) => typeof c[f] !== 'string'))) {
    problems.push('suggested_code malformed');
  }
  return problems;
}

// Promptfoo may pass an array var as a JSON string, so accept either form.
function toList(v) {
  if (Array.isArray(v)) return v;
  try { const p = JSON.parse(v); if (Array.isArray(p)) return p; } catch (e) { /* not JSON */ }
  return String(v).split(',').map((x) => x.trim()).filter(Boolean);
}

const norm = (c) => String(c).toUpperCase().replace(/[^A-Z0-9]/g, '');

// Reply must be bare JSON (a markdown fence fails) and match the schema.
function strictValid(output) {
  let obj;
  try { obj = JSON.parse(output.trim()); } catch (e) {
    return { pass: false, score: 0, reason: 'not bare JSON (fence, prose, or invalid)' };
  }
  const p = schemaProblems(obj);
  return { pass: p.length === 0, score: p.length === 0 ? 1 : 0, reason: p.join('; ') || 'ok' };
}

// Same schema check after stripping fences / surrounding prose.
function lenientValid(output) {
  const obj = parseLenient(output);
  if (!obj) return { pass: false, score: 0, reason: 'no JSON object found' };
  const p = schemaProblems(obj);
  return { pass: p.length === 0, score: p.length === 0 ? 1 : 0, reason: p.join('; ') || 'ok' };
}

// Any predicted code shares its first 3 characters with a gold code.
function codeCategoryHit(output, context) {
  const obj = parseLenient(output);
  const pred = obj && Array.isArray(obj.suggested_codes)
    ? obj.suggested_codes.map((c) => norm(c && c.code)).filter(Boolean) : [];
  const gold = toList(context.vars.gold_codes).map(norm);
  const hit = pred.some((p) => gold.some((g) => g.slice(0, 3) === p.slice(0, 3)));
  return { pass: hit, score: hit ? 1 : 0, reason: `gold ${gold.join(',')} vs predicted ${pred.join(',') || 'none'}` };
}

module.exports = { strictValid, lenientValid, codeCategoryHit };
