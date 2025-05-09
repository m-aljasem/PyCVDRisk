// Run the official ACC CVD Risk Estimator+ formula code (tools.acc.org/CVD-Risk-Estimator-Plus,
// assets/js/riskformulas.js, saved unmodified in source/) on fixtures/patients.csv.
// Only its `appmodel.Form()` getters are stubbed. Writes fixtures/acc_reference.csv (unrounded risk).
// Run from repo root: node reference/pce/run_acc.js
const fs = require("fs");
const vm = require("vm");
const dir = "reference/pce";
const ctx = {};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(`${dir}/source/acc_cvd_risk_estimator_riskformulas.js`, "utf8"), ctx);
const [head, ...rows] = fs.readFileSync(`${dir}/fixtures/patients.csv`, "utf8").trim().split("\n");
const cols = head.split(",");
const out = ["id,risk"];
for (const line of rows) {
  const p = Object.fromEntries(line.split(",").map((v, i) => [cols[i], v]));
  const form = {
    Age: () => +p.age, TotalCholesterol: () => +p.total_chol_mgdl, HDLCholesterol: () => +p.hdl_chol_mgdl,
    BloodPressure: () => +p.sbp, isHypertension: () => p.bp_treated === "1", isSmoker: () => p.smoker === "1",
    isDiabetic: () => p.diabetes === "1", isAfrican: () => p.race === "african_american",
    isFemale: () => p.sex === "female", isMale: () => p.sex === "male",
  };
  ctx.appmodel = { Form: () => form, isFollowUp: () => false };
  const f = vm.runInContext("new Formula(appmodel)", ctx);
  out.push(`${p.id},${f.cvdPredict(true).toPrecision(17)}`);
}
fs.writeFileSync(`${dir}/fixtures/acc_reference.csv`, out.join("\n") + "\n");
console.log(`wrote ${rows.length} rows`);
