import { readFileSync } from "node:fs";

// ---- The contract: what one catalog entry looks like ----
export interface TensorInfo {
  dtype: string;
  shape: number[];
  data_offsets: [number, number]; // always exactly [start, end], relative to the data section
}

// ---- What the engine receives: a shape + the numbers ----
export interface Tensor {
  name: string;
  shape: number[];
  data: Float32Array;
}

// ---- bf16 -> fp32: bf16 is fp32's top 16 bits, so glue 16 zero bits underneath ----
function bf16ToF32(bytes: Buffer, start: number, count: number): Float32Array {
  const out = new Float32Array(count);
  const outBits = new Uint32Array(out.buffer); // second lens over the SAME memory
  for (let i = 0; i < count; i++) {
    const u16 = bytes.readUInt16LE(start + i * 2);
    outBits[i] = u16 << 16;
  }
  return out;
}

// ---- The loading dock: file on disk -> named tensors in memory ----
export class SafetensorsFile {
  private readonly buf: Buffer;
  private readonly catalog = new Map<string, TensorInfo>();
  private readonly dataStart: number;

  constructor(path: string) {
    this.buf = readFileSync(path);
    const headerLen = Number(this.buf.readBigUInt64LE(0));
    const header = JSON.parse(this.buf.subarray(8, 8 + headerLen).toString("utf8"));
    for (const [name, info] of Object.entries(header)) {
      if (name === "__metadata__") continue;
      this.catalog.set(name, info as TensorInfo);
    }
    this.dataStart = 8 + headerLen;
  }

  tensorNames(): string[] {
    return [...this.catalog.keys()];
  }

  info(name: string): TensorInfo {
    const t = this.catalog.get(name);
    if (!t) throw new Error(`no tensor named "${name}"`);
    return t;
  }

  getTensor(name: string): Tensor {
    const t = this.info(name);
    if (t.dtype !== "BF16") throw new Error(`${name}: unsupported dtype ${t.dtype}`);
    const [start, end] = t.data_offsets;
    const data = bf16ToF32(this.buf, this.dataStart + start, (end - start) / 2);
    return { name, shape: t.shape, data };
  }

  // Every byte of the data section must be claimed by exactly the catalog: last end offset == file end
  integrityCheck(): boolean {
    let maxEnd = 0;
    for (const t of this.catalog.values()) maxEnd = Math.max(maxEnd, t.data_offsets[1]);
    return this.dataStart + maxEnd === this.buf.length;
  }
}
