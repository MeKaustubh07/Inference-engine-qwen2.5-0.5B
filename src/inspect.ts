import { readFileSync } from "node:fs";
import { SafetensorsFile } from "./safetensors.js";

const MODEL_DIR = "models/qwen2.5-0.5b";
const config = JSON.parse(readFileSync(`${MODEL_DIR}/config.json`).toString("utf8"));
const file = new SafetensorsFile(`${MODEL_DIR}/model.safetensors`);

const names = file.tensorNames();
console.log("tensor count:", names.length);

// Check 1: distinct layer numbers === num_hidden_layers
const layerNums = new Set<number>();
for (const n of names) {
  const m = n.match(/^model\.layers\.(\d+)\./);
  if (m) layerNums.add(Number(m[1]));
}
console.log("check 1 (layers):", layerNums.size === config.num_hidden_layers ? "PASS" : "FAIL");

// Check 2: embedding shape === [vocab_size, hidden_size]
const emb = file.info("model.embed_tokens.weight");
const shapeOk = emb.shape[0] === config.vocab_size && emb.shape[1] === config.hidden_size;
console.log("check 2 (embed shape):", shapeOk ? "PASS" : "FAIL");

// Check 3: every dtype is BF16
const allBf16 = names.every(n => file.info(n).dtype === "BF16");
console.log("check 3 (all BF16):", allBf16 ? "PASS" : "FAIL");

// Check 4 (W1.4): the catalog accounts for every byte of the file
console.log("check 4 (integrity):", file.integrityCheck() ? "PASS" : "FAIL");

// Load one real tensor through the public doorway
const norm = file.getTensor("model.layers.0.input_layernorm.weight");
console.log("norm shape:", norm.shape, "first 4:", Array.from(norm.data.subarray(0, 4)));

// Load the biggest one and time it
const t0 = performance.now();
const embT = file.getTensor("model.embed_tokens.weight");
console.log(`embed shape: [${embT.shape}] elements: ${embT.data.length} loaded in ${(performance.now() - t0).toFixed(0)} ms`);
