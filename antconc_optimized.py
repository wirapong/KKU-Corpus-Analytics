from __future__ import annotations

"""AntConc-like corpus toolkit with multilingual/Thai-aware tokenization.

Features
- Corpus loading from uploaded .txt files
- Word list, concordance (KWIC), concordance plot, file view
- N-grams, collocates, keyword list
- Exact / wildcard / regex matching
- Thai-aware tokenization via PyThaiNLP when available
- Pandas DataFrame outputs for all core analyses
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Pattern, Sequence, Union

import pandas as pd

try:
    from pythainlp.tokenize import word_tokenize as thai_word_tokenize
except Exception:  # pragma: no cover
    thai_word_tokenize = None


DEFAULT_STOPWORDS: set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "here",
    "him", "his", "i", "if", "in", "into", "is", "it", "its", "me", "more", "most", "my",
    "no", "not", "of", "on", "or", "our", "she", "so", "than", "that", "the", "their",
    "them", "there", "these", "they", "this", "those", "to", "under", "up", "us", "was",
    "we", "were", "what", "when", "where", "which", "who", "will", "with", "would", "you",
    "your",
}

THAI_STOPWORDS: set[str] = {
    "การ", "กับ", "ก็", "ก่อน", "กว่า", "กะ", "กัน", "ของ", "ขอ", "ขณะ", "เข้า", "ขึ้น", "คง", "คงจะ",
    "คือ", "ค่ะ", "ครับ", "ครั้ง", "ความ", "ค่อย", "จัง", "จัด", "จาก", "จึง", "จะ", "จังหวัด", "จริง",
    "ช่าง", "ช่วง", "ซึ่ง", "ด้วย", "ด้าน", "ดัง", "ดังกล่าว", "ตั้ง", "ตั้งแต่", "ตาม", "ต่อ", "ต่าง",
    "ต้อง", "ถ้า", "ถึง", "ทั้ง", "ทั้งนี้", "ทาง", "ที่", "ที่สุด", "ทุก", "ทำ", "ทำให้", "ทางด้าน",
    "นั้น", "นอกจากนี้", "นัก", "นับ", "นาน", "นำ", "นี้", "นึง", "หนึ่ง", "น้อย", "บาง", "บางส่วน", "บ้าง",
    "ปัจจุบัน", "ประกอบ", "ประมาณ", "ปรากฏ", "เป็น", "เปิด", "ผู้", "ผู้ที่", "ผ่าน", "เพื่อ", "เพราะ", "เพียง",
    "ภายใต้", "มาก", "มากกว่า", "มี", "มิ", "มิได้", "ย่อม", "ยัง", "รวม", "รวมทั้ง", "ระหว่าง", "ระดับ",
    "รายละเอียด", "รูปแบบ", "ร่วม", "ละ", "ล้วน", "ล่าสุด", "วัน", "ว่า", "สรุป", "ส่วน", "ส่วนใหญ่",
    "สอดคล้อง", "สามารถ", "สำหรับ", "สิ่ง", "สุด", "หลัง", "หลาย", "หรือ", "อย่าง", "อย่างมาก",
    "อย่างไรก็ตาม", "อยาก", "อยู่", "อาจ", "อีก", "เอง", "เอา", "แล้ว", "แบบ", "ให้", "ได้", "ไป", "ไม่",
    "ไว้", "ใน", "โดย", "โดยเฉพาะ", "แห่ง", "แต่", "แต่ละ", "แม้", "แม้ว่า", "และ", "หรือไม่",
    "ครับผม", "คะ", "นะ", "นะคะ", "นะครับ", "จ้า", "เช่น", "เช่นกัน", "เช่นเดียวกัน", "ดังนั้น", "อัน", "อันที่จริง",
    "เรื่อง", "เกี่ยวกับ", "ภายหลัง", "ภายใน", "ภายนอก", "เพิ่ง", "เพิ่งจะ", "เท่านั้น", "แทบ", "แทน", "ตรง", "ต่อไป",
    "ตลอด", "ตลอดจน", "ทีเดียว", "ทุกคน", "ทุกครั้ง", "ทุกวัน", "บางครั้ง", "บางที", "บัดนี้", "เบื้องต้น", "ประการ", "ประเภท",
    "ผล", "พร้อม", "พร้อมทั้ง", "ภาค", "ภายหน้า", "ยิ่ง", "ยิ่งขึ้น", "รวมถึง", "รวมไปถึง", "รวมกัน", "ระบุ", "รวมแล้ว",
    "ลักษณะ", "ล้วนแต่", "สู่", "เสีย", "เสียก่อน", "หน่อย", "หนึ่งๆ", "เหตุ", "เหตุผล", "เห็น", "เห็นว่า", "แสดง", "แห่งนี้",
}

MULTILINGUAL_STOPWORDS: set[str] = DEFAULT_STOPWORDS | THAI_STOPWORDS

REGEX_PRESETS: dict[str, str] = {
    "Starts with": r"^TERM",
    "Ends with": r"TERM$",
    "Contains": r"TERM",
    "Whole word": r"\bTERM\b",
    "Prefix + suffix": r"^TERM.*suffix$",
    "Contains digit": r".*\d.*",
    "Repeated char (e.g., sooo)": r"(.)\1{2,}",
    "Thai numerals": r"[๐-๙]+",
    "Email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "URL": r"https?://\S+|www\.\S+",
    "Thai whole word": r"(?<![ก-๙])TERM(?![ก-๙])",
    "Word family (simple)": r"\bTERM(?:s|ed|ing)?\b",
}

REGEX_REPLACE_PRESETS: dict[str, tuple[str, str]] = {
    "Escape regex special chars": (r"([.^$*+?{}\[\]\|()])", r"\\\1"),
    "Wildcard * -> regex": (r"\*", r".*"),
    "Wildcard ? -> regex": (r"\?", r"."),
    "Space -> optional whitespace": (r"\s+", r"\\s+"),
    "Thai spacing tolerant": (r"\s+", r"\\s*"),
}



@dataclass(slots=True)
class AnalyzerConfig:
    lowercase: bool = True
    language: str = "auto"  # auto|english|thai|multilingual
    thai_engine: str = "newmm"
    kwic_window: int = 5
    collocate_window: int = 5
    ngram_size: int = 2
    keyword_smoothing: float = 0.5
    max_kwic_rows: Optional[int] = 5000
    token_pattern_latin: str = r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?"


@dataclass(slots=True)
class CorpusDocument:
    doc_id: int
    name: str
    path: Optional[str]
    text: str
    tokens: list[str] = field(default_factory=list)
    token_spans: list[tuple[int, int]] = field(default_factory=list)


class CorpusAnalysisError(Exception):
    pass


class CorpusAnalyzer:
    def __init__(self, config: Optional[AnalyzerConfig] = None) -> None:
        self.config = config or AnalyzerConfig()
        self.documents: list[CorpusDocument] = []
        self._latin_re: Pattern[str] = re.compile(self.config.token_pattern_latin)
        self._mixed_re: Pattern[str] = re.compile(
            r"[A-Za-z]+(?:'[A-Za-z]+)?|[ก-๙]+|\d+(?:\.\d+)?",
            flags=re.UNICODE,
        )
        self._token_table: Optional[pd.DataFrame] = None

    @classmethod
    def from_files(
        cls,
        paths: Union[str, Path, Sequence[Union[str, Path]]],
        config: Optional[AnalyzerConfig] = None,
        recursive: bool = False,
        suffixes: Sequence[str] = (".txt",),
    ) -> "CorpusAnalyzer":
        analyzer = cls(config=config)
        analyzer.load_files(paths=paths, recursive=recursive, suffixes=suffixes)
        return analyzer

    def load_files(
        self,
        paths: Union[str, Path, Sequence[Union[str, Path]]],
        recursive: bool = False,
        suffixes: Sequence[str] = (".txt",),
    ) -> "CorpusAnalyzer":
        file_paths = self._expand_paths(paths, recursive=recursive, suffixes=suffixes)
        if not file_paths:
            raise CorpusAnalysisError("No matching .txt files were found.")
        errors: list[str] = []
        for file_path in file_paths:
            try:
                text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
                self.add_text(text=text, name=Path(file_path).name, path=str(file_path))
            except Exception as exc:
                errors.append(f"{file_path}: {exc}")
        if not self.documents:
            raise CorpusAnalysisError("Unable to load corpus.\n" + "\n".join(errors))
        return self

    def add_text(self, text: str, name: str = "document", path: Optional[str] = None) -> "CorpusAnalyzer":
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        normalized = text.lower() if self.config.lowercase else text
        tokens, spans = self._tokenize_with_spans(normalized)
        self.documents.append(
            CorpusDocument(
                doc_id=len(self.documents),
                name=name,
                path=path,
                text=normalized,
                tokens=tokens,
                token_spans=spans,
            )
        )
        self._token_table = None
        return self

    def basic_stats(self) -> pd.DataFrame:
        self._ensure_corpus()
        rows = []
        for doc in self.documents:
            total = len(doc.tokens)
            unique = len(set(doc.tokens))
            rows.append({
                "doc_id": doc.doc_id,
                "document": doc.name,
                "total_words": total,
                "unique_words": unique,
                "sentences": len(re.findall(r"[.!?]+|[।]+", doc.text)) or (1 if doc.text.strip() else 0),
                "avg_word_length": round((sum(map(len, doc.tokens)) / total), 3) if total else 0.0,
                "type_token_ratio": round(unique / total, 4) if total else 0.0,
            })
        return pd.DataFrame(rows).sort_values("document").reset_index(drop=True)

    def word_list(self, min_freq: int = 1, stopwords: Optional[set[str]] = None, keep_numbers: bool = True) -> pd.DataFrame:
        tokens = self._token_table_frame().copy()
        if tokens.empty:
            return pd.DataFrame(columns=["rank", "token", "frequency", "relative_freq_per_million"])
        if not keep_numbers:
            tokens = tokens[~tokens["token"].str.fullmatch(r"\d+(?:\.\d+)?", na=False)]
        if stopwords:
            tokens = tokens[~tokens["token"].isin(stopwords)]
        freq = tokens.groupby("token").size().reset_index(name="frequency")
        freq = freq[freq["frequency"] >= int(min_freq)]
        freq = freq.sort_values(["frequency", "token"], ascending=[False, True]).reset_index(drop=True)
        total_tokens = len(tokens)
        freq["rank"] = range(1, len(freq) + 1)
        freq["relative_freq_per_million"] = (freq["frequency"] / max(total_tokens, 1)) * 1_000_000
        return freq[["rank", "token", "frequency", "relative_freq_per_million"]]

    def concordance(
        self,
        query: str,
        window: Optional[int] = None,
        regex: bool = False,
        wildcard: bool = False,
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        self._ensure_corpus()
        matcher = self._build_query_matcher(query=query, regex=regex, wildcard=wildcard)
        window = self.config.kwic_window if window is None else int(window)
        max_rows = self.config.max_kwic_rows if max_rows is None else max_rows
        rows: list[dict[str, Any]] = []
        token_table = self._token_table_frame()
        for doc in self.documents:
            dt = token_table[token_table["doc_id"] == doc.doc_id].reset_index(drop=True)
            if dt.empty:
                continue
            for pos, token in enumerate(dt["token"].tolist()):
                matched_pattern = matcher(token)
                if not matched_pattern:
                    continue
                left_tokens = dt.loc[max(0, pos - window): pos - 1, "token"].tolist()
                right_tokens = dt.loc[pos + 1: pos + window, "token"].tolist()
                left_context = " ".join(left_tokens)
                right_context = " ".join(right_tokens)
                kwic_plain = f"{left_context} <<{token}>> {right_context}".strip()
                rows.append({
                    "doc_id": doc.doc_id,
                    "document": doc.name,
                    "token_index": int(dt.loc[pos, "token_index"]),
                    "char_start": int(dt.loc[pos, "char_start"]),
                    "char_end": int(dt.loc[pos, "char_end"]),
                    "left_context": left_context,
                    "keyword": token,
                    "right_context": right_context,
                    "kwic": kwic_plain,
                    "matched_pattern": str(matched_pattern),
                })
                if max_rows is not None and len(rows) >= max_rows:
                    return pd.DataFrame(rows)
        return pd.DataFrame(rows)

    def concordance_plot(self, query: str, regex: bool = False, wildcard: bool = False) -> pd.DataFrame:
        self._ensure_corpus()
        matcher = self._build_query_matcher(query=query, regex=regex, wildcard=wildcard)
        rows = []
        token_table = self._token_table_frame()
        for doc in self.documents:
            dt = token_table[token_table["doc_id"] == doc.doc_id].reset_index(drop=True)
            total = len(dt)
            if total == 0:
                continue
            for pos, token in enumerate(dt["token"].tolist()):
                matched_pattern = matcher(token)
                if matched_pattern:
                    rows.append({
                        "doc_id": doc.doc_id,
                        "document": doc.name,
                        "token_index": int(dt.loc[pos, "token_index"]),
                        "position_pct": round((pos / total) * 100, 3),
                        "keyword": token,
                        "char_start": int(dt.loc[pos, "char_start"]),
                        "char_end": int(dt.loc[pos, "char_end"]),
                        "matched_pattern": str(matched_pattern),
                    })
        return pd.DataFrame(rows)

    def file_view(
        self,
        query: str,
        regex: bool = False,
        wildcard: bool = False,
        context_chars: int = 120,
        max_rows: int = 200,
    ) -> pd.DataFrame:
        self._ensure_corpus()
        matcher = self._build_query_matcher(query=query, regex=regex, wildcard=wildcard)
        rows = []
        for doc in self.documents:
            for token, (start, end) in zip(doc.tokens, doc.token_spans):
                matched_pattern = matcher(token)
                if matched_pattern:
                    row = self._file_view_row(doc, token, start, end, context_chars=context_chars)
                    row["matched_pattern"] = str(matched_pattern)
                    rows.append(row)
                    if len(rows) >= max_rows:
                        return pd.DataFrame(rows)
        return pd.DataFrame(rows)

    def file_view_at(self, doc_id: int, char_start: int, char_end: int, context_chars: int = 120) -> pd.DataFrame:
        self._ensure_corpus()
        doc = self.documents[int(doc_id)]
        token = doc.text[char_start:char_end]
        return pd.DataFrame([self._file_view_row(doc, token, int(char_start), int(char_end), context_chars=context_chars)])

    def ngrams(self, n: Optional[int] = None, min_freq: int = 2, stopwords: Optional[set[str]] = None) -> pd.DataFrame:
        self._ensure_corpus()
        n = self.config.ngram_size if n is None else int(n)
        if n < 2:
            raise ValueError("n must be >= 2")
        counter: Counter[tuple[str, ...]] = Counter()
        for doc in self.documents:
            toks = doc.tokens
            for i in range(len(toks) - n + 1):
                gram = tuple(toks[i:i+n])
                if stopwords and all(tok in stopwords for tok in gram):
                    continue
                counter[gram] += 1
        if not counter:
            return pd.DataFrame(columns=["rank", "ngram", "frequency"])
        df = pd.DataFrame([
            {"ngram": " ".join(k), "frequency": v}
            for k, v in counter.items() if v >= int(min_freq)
        ])
        if df.empty:
            return pd.DataFrame(columns=["rank", "ngram", "frequency"])
        df = df.sort_values(["frequency", "ngram"], ascending=[False, True]).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df[["rank", "ngram", "frequency"]]

    def collocates(
        self,
        query: str,
        window: Optional[int] = None,
        min_freq: int = 2,
        stopwords: Optional[set[str]] = None,
        regex: bool = False,
        wildcard: bool = False,
    ) -> pd.DataFrame:
        self._ensure_corpus()
        matcher = self._build_query_matcher(query=query, regex=regex, wildcard=wildcard)
        window = self.config.collocate_window if window is None else int(window)
        tt = self._token_table_frame()
        token_freq = tt["token"].value_counts()
        corpus_size = len(tt)
        co_counts: Counter[str] = Counter()
        target_count = 0
        for doc in self.documents:
            toks = doc.tokens
            hit_positions = [i for i, tok in enumerate(toks) if matcher(tok)]
            target_count += len(hit_positions)
            for pos in hit_positions:
                left = max(0, pos - window)
                right = min(len(toks), pos + window + 1)
                for word in toks[left:pos] + toks[pos+1:right]:
                    if stopwords and word in stopwords:
                        continue
                    co_counts[word] += 1
        if target_count == 0:
            return pd.DataFrame(columns=["rank", "collocate", "co_freq", "freq", "pmi", "t_score"])
        rows = []
        for collocate, co_freq in co_counts.items():
            if co_freq < int(min_freq):
                continue
            freq = int(token_freq.get(collocate, 0))
            expected = (target_count * freq) / max(corpus_size, 1)
            pmi = math.log2(co_freq / expected) if expected > 0 and co_freq > 0 else 0.0
            t_score = (co_freq - expected) / math.sqrt(co_freq) if co_freq > 0 else 0.0
            rows.append({
                "collocate": collocate,
                "co_freq": int(co_freq),
                "freq": freq,
                "pmi": round(pmi, 4),
                "t_score": round(t_score, 4),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["rank", "collocate", "co_freq", "freq", "pmi", "t_score"])
        df = df.sort_values(["co_freq", "pmi", "collocate"], ascending=[False, False, True]).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df[["rank", "collocate", "co_freq", "freq", "pmi", "t_score"]]

    def keyword_list(self, reference: Union["CorpusAnalyzer", pd.DataFrame, dict[str, int]], min_freq: int = 2) -> pd.DataFrame:
        self._ensure_corpus()
        target_df = self.word_list(min_freq=1)[["token", "frequency"]].rename(columns={"frequency": "target_freq"})
        ref_df = self._reference_to_freq_df(reference).rename(columns={"frequency": "reference_freq"})
        merged = target_df.merge(ref_df, on="token", how="outer").fillna(0)
        n1 = float(target_df["target_freq"].sum())
        n2 = float(ref_df["reference_freq"].sum())
        if n1 == 0 or n2 == 0:
            raise CorpusAnalysisError("Both target and reference corpora must contain tokens.")
        merged["log_likelihood"] = merged.apply(lambda r: self._log_likelihood(float(r["target_freq"]), float(r["reference_freq"]), n1, n2), axis=1)
        merged["effect"] = ((merged["target_freq"] + self.config.keyword_smoothing) / n1) / ((merged["reference_freq"] + self.config.keyword_smoothing) / n2)
        merged["keyness_direction"] = merged.apply(lambda r: "target" if (r["target_freq"] / n1) >= (r["reference_freq"] / n2) else "reference", axis=1)
        merged = merged[(merged["target_freq"] >= int(min_freq)) | (merged["reference_freq"] >= int(min_freq))]
        merged = merged.sort_values(["log_likelihood", "effect", "token"], ascending=[False, False, True]).reset_index(drop=True)
        merged["rank"] = range(1, len(merged) + 1)
        return merged[["rank", "token", "target_freq", "reference_freq", "log_likelihood", "effect", "keyness_direction"]]

    def _token_table_frame(self) -> pd.DataFrame:
        if self._token_table is not None:
            return self._token_table
        rows = []
        for doc in self.documents:
            for i, (token, span) in enumerate(zip(doc.tokens, doc.token_spans)):
                rows.append({
                    "doc_id": doc.doc_id,
                    "document": doc.name,
                    "token_index": i,
                    "token": token,
                    "char_start": span[0],
                    "char_end": span[1],
                })
        self._token_table = pd.DataFrame(rows)
        return self._token_table

    def _tokenize_with_spans(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        mode = self._resolve_language_mode(text)
        if mode == "thai":
            return self._tokenize_thai(text)
        if mode == "multilingual":
            return self._tokenize_mixed(text)
        return self._tokenize_latin(text)

    def _tokenize_latin(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        toks, spans = [], []
        for m in self._latin_re.finditer(text):
            toks.append(m.group(0))
            spans.append(m.span())
        return toks, spans

    def _tokenize_mixed(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        toks, spans = [], []
        for m in self._mixed_re.finditer(text):
            segment = m.group(0)
            start = m.start()
            if re.fullmatch(r"[ก-๙]+", segment) and thai_word_tokenize is not None:
                offset = start
                for tok in thai_word_tokenize(segment, engine=self.config.thai_engine, keep_whitespace=False):
                    if not tok.strip():
                        continue
                    rel = segment.find(tok, max(0, offset - start))
                    if rel == -1:
                        rel = 0
                    s = start + rel
                    e = s + len(tok)
                    toks.append(tok)
                    spans.append((s, e))
                    offset = e
            else:
                toks.append(segment)
                spans.append(m.span())
        return toks, spans

    def _tokenize_thai(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        if thai_word_tokenize is None:
            return self._tokenize_mixed(text)
        toks, spans = [], []
        last_end = 0
        for m in re.finditer(r"[ก-๙]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?", text):
            segment = m.group(0)
            start = m.start()
            if re.fullmatch(r"[ก-๙]+", segment):
                cursor = start
                consumed = 0
                for tok in thai_word_tokenize(segment, engine=self.config.thai_engine, keep_whitespace=False):
                    if not tok.strip():
                        continue
                    rel = segment.find(tok, consumed)
                    if rel == -1:
                        rel = consumed
                    s = start + rel
                    e = s + len(tok)
                    toks.append(tok)
                    spans.append((s, e))
                    consumed = rel + len(tok)
                    cursor = e
                last_end = cursor
            else:
                toks.append(segment)
                spans.append(m.span())
                last_end = m.end()
        return toks, spans

    def _resolve_language_mode(self, text: str) -> str:
        mode = (self.config.language or "auto").lower()
        if mode != "auto":
            return mode
        thai_chars = len(re.findall(r"[ก-๙]", text))
        latin_chars = len(re.findall(r"[A-Za-z]", text))
        if thai_chars and latin_chars:
            return "multilingual"
        if thai_chars:
            return "thai"
        return "english"

    @staticmethod
    def _split_multi_patterns(query: str) -> list[str]:
        raw_parts = re.split(r"[,;\n]+", str(query))
        parts = [p.strip() for p in raw_parts if p.strip()]
        return parts or [str(query).strip()]

    def kwic_sort(self, df: pd.DataFrame, sort_by: str = "position") -> pd.DataFrame:
        if df is None or df.empty or sort_by in {"position", "none"}:
            return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
        out = df.copy()
        for i in range(1, 6):
            out[f"L{i}"] = out["left_context"].fillna("").str.split().apply(lambda xs, n=i: xs[-n] if len(xs) >= n else "")
            out[f"R{i}"] = out["right_context"].fillna("").str.split().apply(lambda xs, n=i: xs[n-1] if len(xs) >= n else "")
        key = sort_by.upper()
        if key.startswith("L") or key.startswith("R"):
            out = out.sort_values([key, "keyword", "document", "token_index"], kind="stable").reset_index(drop=True)
        return out

    def _build_query_matcher(self, query: str, regex: bool = False, wildcard: bool = False) -> Callable[[str], Optional[str]]:
        if not query or not str(query).strip():
            raise ValueError("query must not be empty")
        flags = 0 if self.config.lowercase else re.IGNORECASE
        parts = self._split_multi_patterns(query)
        if regex:
            patterns = [(part, re.compile(part, flags=flags)) for part in parts]
            return lambda tok: next((raw for raw, pattern in patterns if pattern.search(tok)), None)
        if wildcard:
            patterns = [(part, re.compile(self._wildcard_to_regex(part), flags=flags)) for part in parts]
            return lambda tok: next((raw for raw, pattern in patterns if pattern.fullmatch(tok)), None)
        exacts = {part.lower() if self.config.lowercase else part: part for part in parts}
        return lambda tok: exacts.get(tok, None)

    @staticmethod
    def _wildcard_to_regex(pattern: str) -> str:
        escaped = re.escape(pattern)
        escaped = escaped.replace(r"\*", ".*").replace(r"\?", ".")
        return f"^{escaped}$"

    def _file_view_row(self, doc: CorpusDocument, token: str, start: int, end: int, context_chars: int) -> dict[str, Any]:
        s = max(0, int(start) - context_chars)
        e = min(len(doc.text), int(end) + context_chars)
        return {
            "doc_id": doc.doc_id,
            "document": doc.name,
            "keyword": token,
            "char_start": int(start),
            "char_end": int(end),
            "snippet": doc.text[s:e].replace("\n", " "),
        }

    def _reference_to_freq_df(self, reference: Union["CorpusAnalyzer", pd.DataFrame, dict[str, int]]) -> pd.DataFrame:
        if isinstance(reference, CorpusAnalyzer):
            return reference.word_list(min_freq=1)[["token", "frequency"]]
        if isinstance(reference, dict):
            return pd.DataFrame({"token": list(reference.keys()), "frequency": list(reference.values())})
        if isinstance(reference, pd.DataFrame):
            cols = set(reference.columns)
            if {"token", "frequency"}.issubset(cols):
                return reference[["token", "frequency"]].copy()
        raise TypeError("reference must be CorpusAnalyzer, DataFrame(token, frequency), or dict[str, int]")

    @staticmethod
    def _log_likelihood(a: float, b: float, N1: float, N2: float) -> float:
        e1 = N1 * (a + b) / (N1 + N2)
        e2 = N2 * (a + b) / (N1 + N2)
        term1 = a * math.log(a / e1) if a > 0 and e1 > 0 else 0.0
        term2 = b * math.log(b / e2) if b > 0 and e2 > 0 else 0.0
        return round(2 * (term1 + term2), 6)

    def _expand_paths(self, paths: Union[str, Path, Sequence[Union[str, Path]]], recursive: bool, suffixes: Sequence[str]) -> list[Path]:
        if isinstance(paths, (str, Path)):
            raw = [Path(paths)]
        else:
            raw = [Path(p) for p in paths]
        result: list[Path] = []
        allowed = {s.lower() for s in suffixes}
        for p in raw:
            if p.is_file() and p.suffix.lower() in allowed:
                result.append(p)
            elif p.is_dir():
                pattern = "**/*" if recursive else "*"
                result.extend([f for f in p.glob(pattern) if f.is_file() and f.suffix.lower() in allowed])
        return sorted(set(result))

    def _ensure_corpus(self) -> None:
        if not self.documents:
            raise CorpusAnalysisError("Corpus is empty. Please load one or more .txt files first.")
