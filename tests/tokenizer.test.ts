import { readFileSync } from "node:fs";
import { Tokenizer } from "../src/tokenizer.js";

const tok = new Tokenizer("models/qwen2.5-0.5b/tokenizer.json");
const cases = JSON.parse(readFileSync("tests/golden_tokens.json").toString("utf8")) as { text: string; ids: number[] }[];

let pass = 0;
for (const c of cases) {
  const ids = tok.encode(c.text);
  const same = ids.length === c.ids.length && ids.every((v, i) => v === c.ids[i]);
  const roundTrip = tok.decode(ids) === c.text;
  if (same && roundTrip) pass++;
  else console.log("FAIL", JSON.stringify(c.text), "\n  ours:", ids, "\n  gold:", c.ids, "\n  roundtrip ok:", roundTrip);
}
console.log(`${pass}/${cases.length} cases match the official tokenizer`);
