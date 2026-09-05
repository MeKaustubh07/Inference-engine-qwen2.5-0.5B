import { readFileSync } from "node:fs";

// ============================================================
// Stage 2b helper: the 256-entry byte alphabet.
// Every byte value 0..255 gets a printable stand-in character.
// Printable ASCII/Latin bytes map to themselves; the rest are
// shifted up to 256+ so they become visible, unique characters.
// (This is GPT-2's bytes_to_unicode, which Qwen inherits.)
// ============================================================
function buildByteAlphabet(): { byteToChar: string[]; charToByte: Map<string, number> } {
  const printable: number[] = [];
  for (let b = 33; b <= 126; b++) printable.push(b);   // '!' .. '~'
  for (let b = 161; b <= 172; b++) printable.push(b);  // '¡' .. '¬'
  for (let b = 174; b <= 255; b++) printable.push(b);  // '®' .. 'ÿ'

  const byteToChar: string[] = new Array(256);
  let next = 256;
  for (let b = 0; b < 256; b++) {
    if (printable.includes(b)) byteToChar[b] = String.fromCharCode(b);
    else byteToChar[b] = String.fromCharCode(next++); // e.g. space (32) -> 'Ġ' (288)
  }
  const charToByte = new Map<string, number>();
  byteToChar.forEach((ch, b) => charToByte.set(ch, b));
  return { byteToChar, charToByte };
}

// ============================================================
// Stage 2 helper: the pre-tokenizer regex (from tokenizer.json),
// with the (?i:...) group rewritten as explicit cases so it runs
// on any JS engine. Flags: g = all matches, u = Unicode \p{...}.
// ============================================================
const PRE_TOKENIZE = new RegExp(
  "'(?:[sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])" +   // contractions
  "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+" +                          // optional non-letter + run of letters
  "|\\p{N}" +                                                 // a single digit
  "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*" +                         // optional space + punctuation run
  "|\\s*[\\r\\n]+" +                                          // newlines
  "|\\s+(?!\\S)" +                                            // whitespace not followed by non-space
  "|\\s+",                                                    // any remaining whitespace
  "gu",
);

export class Tokenizer {
  private readonly vocab = new Map<string, number>();     // token string -> id
  private readonly idToToken: string[] = [];              // id -> token string
  private readonly mergeRank = new Map<string, number>(); // "a b" -> priority (lower = earlier)
  private readonly special = new Map<string, number>();   // "<|im_start|>" -> id
  private readonly specialSplit: RegExp;
  private readonly byteToChar: string[];
  private readonly charToByte: Map<string, number>;

  constructor(path: string) {
    const json = JSON.parse(readFileSync(path).toString("utf8"));

    for (const [tok, id] of Object.entries(json.model.vocab as Record<string, number>)) {
      this.vocab.set(tok, id);
      this.idToToken[id] = tok;
    }
    (json.model.merges as string[]).forEach((m, rank) => this.mergeRank.set(m, rank));

    for (const t of json.added_tokens as { id: number; content: string }[]) {
      this.special.set(t.content, t.id);
      this.idToToken[t.id] = t.content;
    }
    // Regex that finds any special token verbatim, e.g. <\|im_start\|>|<\|im_end\|>|...
    const escaped = [...this.special.keys()].map(s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    this.specialSplit = new RegExp(`(${escaped.join("|")})`, "g");

    ({ byteToChar: this.byteToChar, charToByte: this.charToByte } = buildByteAlphabet());
  }

  // ---------- Stage 3: BPE merging for ONE chunk ----------
  private bpe(chunk: string): string[] {
    let symbols = Array.from(chunk);           // start as single stand-in characters
    while (symbols.length > 1) {
      // find the adjacent pair with the best (lowest) merge rank
      let bestRank = Infinity, bestIdx = -1;
      for (let i = 0; i < symbols.length - 1; i++) {
        const rank = this.mergeRank.get(`${symbols[i]} ${symbols[i + 1]}`);
        if (rank !== undefined && rank < bestRank) { bestRank = rank; bestIdx = i; }
      }
      if (bestIdx === -1) break;               // no mergeable pair left
      // merge every occurrence of that pair
      const merged = symbols[bestIdx] + symbols[bestIdx + 1];
      const out: string[] = [];
      for (let i = 0; i < symbols.length; i++) {
        if (i < symbols.length - 1 && symbols[i] + symbols[i + 1] === merged &&
            this.mergeRank.get(`${symbols[i]} ${symbols[i + 1]}`) === bestRank) {
          out.push(merged); i++;
        } else out.push(symbols[i]);
      }
      symbols = out;
    }
    return symbols;
  }

  // ---------- Encode: text -> ids (all five stages) ----------
  encode(text: string): number[] {
    const ids: number[] = [];
    // Stage 0: peel off special tokens first so they're never split
    for (const part of text.split(this.specialSplit)) {
      if (part === "") continue;
      const sid = this.special.get(part);
      if (sid !== undefined) { ids.push(sid); continue; }

      const normalized = part.normalize("NFC");                       // Stage 1
      for (const chunk of normalized.match(PRE_TOKENIZE) ?? []) {      // Stage 2
        const bytes = Buffer.from(chunk, "utf8");
        let mapped = "";
        for (const b of bytes) mapped += this.byteToChar[b];          // Stage 2b
        for (const piece of this.bpe(mapped)) {                        // Stage 3
          const id = this.vocab.get(piece);
          if (id === undefined) throw new Error(`piece not in vocab: ${JSON.stringify(piece)}`);
          ids.push(id);
        }
      }
    }
    return ids;                                                        // Stage 4: nothing to add
  }

  // ---------- Decode: ids -> text (Stage 5) ----------
  decode(ids: number[]): string {
    const bytes: number[] = [];
    for (const id of ids) {
      const tok = this.idToToken[id];
      if (tok === undefined) throw new Error(`unknown id ${id}`);
      if (this.special.has(tok)) { bytes.push(...Buffer.from(tok, "utf8")); continue; }
      for (const ch of tok) bytes.push(this.charToByte.get(ch)!);   // stand-in char -> byte
    }
    return Buffer.from(bytes).toString("utf8");
  }

  vocabSize(): number { return this.idToToken.length; }
}
