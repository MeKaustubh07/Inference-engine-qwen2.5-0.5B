"""Answer key: tokenize test strings with the official HF tokenizer, write JSON goldens."""
import json, random
from tokenizers import Tokenizer

tok = Tokenizer.from_file("models/qwen2.5-0.5b/tokenizer.json")
cases = [
    "The capital of France is",
    "Hello world",
    "hello   world  ",
    "I'm here, you're there. Don't stop!",
    "Numbers: 12345 and 3.14159, year 2026",
    "Kaustubh builds an inference engine in TypeScript.",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
    "नमस्ते दुनिया — こんにちは世界 — 🚀🔥",
    "def f(x):\n    return x * 2\n\n\n",
    "   leading spaces and\ttabs\tand trailing   ",
    "",
]
random.seed(0)
alphabet = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJ0123456789.,!?'\n-_()[]{}<>|éü你好😀"
for _ in range(40):
    cases.append("".join(random.choice(alphabet) for _ in range(random.randint(1, 60))))

out = [{"text": c, "ids": tok.encode(c).ids} for c in cases]
json.dump(out, open("tests/golden_tokens.json", "w"), ensure_ascii=False, indent=0)
print(f"wrote {len(out)} cases to tests/golden_tokens.json")
