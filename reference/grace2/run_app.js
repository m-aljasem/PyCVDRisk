// Run the developers' GRACE 2.0 calculator code (source/, unmodified) and write its results.
// Run from repo root: node reference/grace2/run_app.js
//
// 1. fixtures/app_reference.csv: the calculator's 1- and 3-year models (grace_lookups.js, the functions
//    commonFunctions.js calls) for the dropdown entries of source/index.html: a base patient, every entry
//    of every input changed one at a time, and 3000 seeded random combinations, each with creatinine and
//    Killip class (full model) and with renal insufficiency and diuretics (substituted model). The
//    functions round their result with Number.prototype.roundup(1); each value is written as shown that
//    way and unrounded (roundup replaced by the identity).
// 2. fixtures/gracejs_reference.csv: grace.js (the 1-year death equations, continuous inputs), unrounded,
//    for 500 seeded random patients.
// 3. fixtures/qunit.json (below).
const fs = require("fs");
const vm = require("vm");
const dir = "reference/grace2";
const ctx = {};
vm.createContext(ctx);
for (const f of ["grace_lookups.js", "grace.js"]) vm.runInContext(fs.readFileSync(`${dir}/source/${f}`, "utf8"), ctx);
const shown = vm.runInContext("Number.prototype.roundup", ctx);
const identity = vm.runInContext("(function () { return this.valueOf(); })", ctx);
// the shipped roundup without its branch that returns Math.round(this) from 10 up
const noBranch = vm.runInContext("(function (dec) { var m = Math.pow(10, dec); return (Math.round(this * m) / m).toFixed(dec); })", ctx);
const setRound = (f) => { ctx.__roundup = f; vm.runInContext("Number.prototype.roundup = __roundup", ctx); };

const html = fs.readFileSync(`${dir}/source/index.html`, "utf8");
const options = (id) => {
  const sel = html.match(new RegExp(`<select[^>]*id="${id}"[^>]*>([\\s\\S]*?)</select>`))[1];
  return [...sel.matchAll(/<option value="([^"]*)"(?! disabled)/g)].map((m) => m[1]).filter((v) => v !== "not");
};
const OPT = {
  age: options("selectAge"), pulse: options("selectHeart"), sbp: options("selectBlood"),
  creatinine: options("selectCreatine").filter((v) => v !== "na"), killip: options("selectCHF").filter((v) => v !== "na"),
  renal: options("selectRenal"), diuretic: options("selectDiuretic"), cardiac: options("selectCardiac"),
  enzymes: options("selectTroponin"), stemi: options("selectSegment"),
};
const MODELS = {
  full: ["creatinine", "killip", [["d1", "CalculateOneYearDeath"], ["dmi1", "CalculateOneYearDeathMI"],
    ["d3", "CalculateThreeYearDeath"], ["dmi3", "CalculateThreeYearDeathMI"]]],
  substituted: ["renal", "diuretic", [["d1", "CalculateOneYearDeathSubstituted"], ["dmi1", "CalculateOneYearDeathSubstitutedMI"],
    ["d3", "CalculateThreeYearDeathSubstituted"], ["dmi3", "CalculateThreeYearDeathSubstitutedMI"]]],
};

let seed = 2014;
const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
const pick = (a) => a[Math.floor(rand() * a.length)];

const rows = [];
for (const [model, [c, k, fns]] of Object.entries(MODELS)) {
  const vars = ["age", "pulse", "sbp", c, k, "cardiac", "enzymes", "stemi"];
  const base = { age: "65", pulse: "84.5", sbp: "124.5", creatinine: "0.995", killip: "1", renal: "no", diuretic: "no",
    cardiac: "no", enzymes: "no", stemi: "no" };
  const ps = [base];
  for (const v of vars) for (const o of OPT[v]) if (o !== base[v]) ps.push({ ...base, [v]: o });
  for (let i = 0; i < 3000; i++) ps.push(Object.fromEntries(vars.map((v) => [v, pick(OPT[v])])));
  for (const p of ps) {
    const args = vars.map((v) => p[v]);
    const r = { model, ...Object.fromEntries(vars.map((v) => [v, p[v]])) };
    for (const [col, fn] of fns) {
      setRound(shown);
      r[`${col}_shown`] = String(ctx[fn](...args));
      setRound(identity);
      r[col] = ctx[fn](...args).toPrecision(17);
    }
    rows.push(r);
  }
}
const cols = ["model", "age", "pulse", "sbp", "creatinine", "killip", "renal", "diuretic", "cardiac", "enzymes", "stemi",
  "d1", "d1_shown", "dmi1", "dmi1_shown", "d3", "d3_shown", "dmi3", "dmi3_shown"];
fs.writeFileSync(`${dir}/fixtures/app_reference.csv`,
  [cols.join(","), ...rows.map((r) => cols.map((c) => r[c] ?? "").join(","))].join("\n") + "\n");

// grace.js: continuous inputs; killip 1-4, flags as booleans
setRound(identity);
const g = ["id,age,pulse,sbp,creatinine,killip,renal,diuretic,cardiac,enzymes,stemi,d1,d1_substituted"];
for (let i = 0; i < 500; i++) {
  const p = [20 + 80 * rand(), 40 + 180 * rand(), 60 + 180 * rand(), 0.2 + 5 * rand(), 1 + Math.floor(4 * rand())]
    .map((x, j) => (j < 4 ? +x.toFixed(2) : x));
  const f = [0, 1, 2, 3, 4].map(() => (rand() < 0.4 ? 1 : 0));
  const [age, pulse, sbp, creat, killip] = p;
  const [renal, diuretic, cardiac, enzymes, stemi] = f;
  const long = ctx.oneyear_longform(age, pulse, sbp, creat, cardiac, enzymes, stemi, killip);
  const sub = ctx.oneyear_substituted(age, pulse, sbp, renal, cardiac, enzymes, stemi, diuretic);
  g.push([i, ...p, ...f, long.toPrecision(17), sub.toPrecision(17)].join(","));
}
fs.writeFileSync(`${dir}/fixtures/gracejs_reference.csv`, g.join("\n") + "\n");

// 3. fixtures/qunit.json: the developers' QUnit tests (source/tests.js) against grace.js, with the roundup the
//    calculator ships and with that roundup minus its >= 10 branch; equal() is QUnit's loose ==.
const qunit = {};
for (const [mode, fn] of [["shipped_roundup", shown], ["roundup_without_ge10_branch", noBranch]]) {
  setRound(fn);
  const failed = new Set();
  let n = 0, bad = 0, name = "";
  ctx.test = (t, f) => { name = t; f(); };
  ctx.equal = (a, b) => { n++; if (!(a == b)) { bad++; failed.add(name); } };
  vm.runInContext(fs.readFileSync(`${dir}/source/tests.js`, "utf8"), ctx);
  qunit[mode] = { asserts: n, failed: bad, failed_tests: [...failed] };
}
fs.writeFileSync(`${dir}/fixtures/qunit.json`, JSON.stringify(qunit, null, 1) + "\n");
console.log(`wrote ${rows.length} calculator rows, 500 grace.js rows, QUnit ${JSON.stringify(qunit)}`);
