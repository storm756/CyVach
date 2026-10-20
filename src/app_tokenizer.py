"""Exact Python port of the app's tokenizer (SMSReaderApp/src/ml/scamDetector.ts).

Differences from Hugging Face BertTokenizer are intentional: this reproduces what
runs on the phone. JS strings iterate UTF-16 code units, so characters outside
the BMP (most emoji) become two lone surrogates, each mapped to [UNK].
"""
from __future__ import annotations

import re
from pathlib import Path

MAX_LEN = 128


class AppTokenizer:
    def __init__(self, vocab_path: str | Path):
        self.tok2id: dict[str, int] = {}
        content = Path(vocab_path).read_text(encoding="utf-8")
        for idx, tok in enumerate(re.split(r"\r?\n", content)):
            if not tok:
                continue
            self.tok2id[tok.strip()] = idx
        self.pad = self.tok2id.get("[PAD]", 0)
        self.unk = self.tok2id.get("[UNK]", 100)
        self.cls = self.tok2id.get("[CLS]", 101)
        self.sep = self.tok2id.get("[SEP]", 102)

    @staticmethod
    def _utf16_units(s: str) -> list[str]:
        out = []
        for ch in s:
            cp = ord(ch)
            if cp > 0xFFFF:
                cp -= 0x10000
                out.append(chr(0xD800 + (cp >> 10)))
                out.append(chr(0xDC00 + (cp & 0x3FF)))
            else:
                out.append(ch)
        return out

    def wordpiece_ids(self, text: str) -> list[int]:
        normalized = text.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)
        tokens: list[str] = []
        cur = ""
        for ch in self._utf16_units(normalized):
            if ("a" <= ch <= "z") or ("0" <= ch <= "9"):
                cur += ch
            else:
                if cur:
                    tokens.append(cur)
                    cur = ""
                if ch.strip():
                    tokens.append(ch)
        if cur:
            tokens.append(cur)
        ids: list[int] = []
        for token in tokens:
            if not token or token.strip() == "":
                continue
            if token in self.tok2id:
                ids.append(self.tok2id[token])
                continue
            start = 0
            sub: list[int] = []
            while start < len(token):
                end = len(token)
                found = False
                while start < end:
                    piece = token[start:end]
                    key = piece if start == 0 else "##" + piece
                    if key in self.tok2id:
                        sub.append(self.tok2id[key])
                        start = end
                        found = True
                        break
                    end -= 1
                if not found:
                    sub.append(self.unk)
                    break
            ids.extend(sub)
        return ids

    def encode(self, text: str, max_len: int = MAX_LEN, pad: bool = True):
        ids = self.wordpiece_ids(text)[: max_len - 2]
        input_ids = [self.cls, *ids, self.sep]
        mask = [1] * len(input_ids)
        if pad:
            while len(input_ids) < max_len:
                input_ids.append(self.pad)
                mask.append(0)
        return input_ids, mask
