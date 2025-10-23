
#!/usr/bin/env python3
"""
IR toolchain (single file):
- Preprocess TREC-like collection (HEADLINE+TEXT): tokenize -> stopwords (provided) -> Porter stem -> positions AFTER stopword removal.
- Build positional inverted index and write in the required text format.
- Boolean / Phrase / Proximity search.
- Ranked retrieval using TFIDF from lecture:
    w_{t,d} = (1 + log10 tf(t,d)) * log10(N / df(t))
    Score(q,d) = sum_{t in q ∩ d} w_{t,d}
"""

import argparse
import html
import io
import json
import math
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Set

# ----------------------------
# Stemming: Porter (NLTK)
# ----------------------------
from nltk.stem import PorterStemmer

# ----------------------------
# Tokeniser (split on any non-letter)
# ----------------------------
TOKEN_PATTERN = re.compile(r"[A-Za-z]+")

def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_PATTERN.finditer(text)]

# ----------------------------
# Stopwords (use provided list; normalise with same tokeniser)
# ----------------------------
def load_stopwords(stop_path: str) -> Set[str]:
    stops: Set[str] = set()
    with io.open(stop_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stops.update(tokenize(line))
    return stops

# ----------------------------
# TREC-like parser (HEADLINE + TEXT)
# ----------------------------
def iter_trec_docs(path: str) -> Iterable[Dict[str, str]]:
    docno = None
    buf_head: List[str] = []
    buf_text: List[str] = []
    in_doc = False
    in_head = False
    in_text = False

    def flush_current():
        nonlocal docno, buf_head, buf_text
        if docno is None:
            return None
        out = {
            "docno": docno,
            "headline": html.unescape("\n".join(buf_head).strip()),
            "text": html.unescape("\n".join(buf_text).strip()),
        }
        docno = None
        buf_head = []
        buf_text = []
        return out

    with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if line.strip() == "<DOC>":
                in_doc = True
                docno = None
                buf_head = []
                buf_text = []
                in_head = False
                in_text = False
                continue
            if line.strip() == "</DOC>":
                in_doc = False
                in_head = False
                in_text = False
                doc = flush_current()
                if doc:
                    yield doc
                continue
            if not in_doc:
                continue

            if line.strip().startswith("<DOCNO>"):
                m = re.search(r"<DOCNO>\s*([^<]+)\s*</DOCNO>", line)
                if m:
                    docno = m.group(1).strip()
                else:
                    content = line
                    while "</DOCNO>" not in content:
                        nxt = next(f, "")
                        content += nxt
                    m2 = re.search(r"<DOCNO>\s*([^<]+)\s*</DOCNO>", content, re.DOTALL)
                    if m2:
                        docno = m2.group(1).strip()
                continue

            if "<HEADLINE>" in line:
                in_head = True
                after = line.split("<HEADLINE>", 1)[1]
                if "</HEADLINE>" in after:
                    before_end, _ = after.split("</HEADLINE>", 1)
                    buf_head.append(before_end)
                    in_head = False
                else:
                    buf_head.append(after)
                continue
            if "</HEADLINE>" in line and in_head:
                before_end, _ = line.split("</HEADLINE>", 1)
                buf_head.append(before_end)
                in_head = False
                continue
            if in_head:
                buf_head.append(line)
                continue

            if "<TEXT>" in line:
                in_text = True
                after = line.split("<TEXT>", 1)[1]
                if "</TEXT>" in after:
                    before_end, _ = after.split("</TEXT>", 1)
                    buf_text.append(before_end)
                    in_text = False
                else:
                    buf_text.append(after)
                continue
            if "</TEXT>" in line and in_text:
                before_end, _ = line.split("</TEXT>", 1)
                buf_text.append(before_end)
                in_text = False
                continue
            if in_text:
                buf_text.append(line)
                continue

# ----------------------------
# Preprocess a single doc
# ----------------------------
def preprocess_doc(doc: Dict[str, str], stopwords: Set[str], stemmer: PorterStemmer) -> Dict:
    headline = doc.get("headline", "") or ""
    text = doc.get("text", "") or ""
    combined = headline + " " + text
    tokens = tokenize(combined)

    processed = []
    pos = 0
    for tok in tokens:
        if tok in stopwords:
            continue
        stem = stemmer.stem(tok)
        pos += 1
        processed.append({"term": stem, "pos": pos})
    return {"docno": doc["docno"], "tokens": processed}

# ----------------------------
# Build positional index from JSONL
# ----------------------------
def build_positional_index(preprocessed_jsonl_path: str) -> dict:
    index: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    with open(preprocessed_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            docno = str(obj["docno"])
            for tok in obj["tokens"]:
                term = tok["term"]
                pos = int(tok["pos"])
                index[term][docno].append(pos)
    for term in index:
        for docno in index[term]:
            index[term][docno].sort()
    return index

def write_index_txt(index: dict, out_path: str) -> None:
    def try_int(x: str):
        try:
            return (0, int(x))
        except:
            return (1, x)
    with open(out_path, "w", encoding="utf-8") as out:
        for term in sorted(index.keys()):
            postings = index[term]
            df = len(postings)
            out.write(f"{term}:{df}\n")
            for docno in sorted(postings.keys(), key=lambda d: try_int(d)):
                pos_str = ", ".join(str(p) for p in postings[docno])
                out.write(f"\t{docno}: {pos_str}\n")

# ----------------------------
# Read index.txt back (for searching)
# ----------------------------
def load_index_from_txt(index_path: str) -> Dict[str, Dict[str, List[int]]]:
    index: Dict[str, Dict[str, List[int]]] = {}
    current_term = None
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if not line.startswith("\t"):
                if ":" in line:
                    term, _df = line.split(":", 1)
                    current_term = term
                    index[current_term] = {}
            else:
                assert current_term is not None, "Malformed index file."
                parts = line.strip().split(":", 1)
                docno = parts[0]
                pos_str = parts[1].strip()
                positions = []
                if pos_str:
                    positions = [int(p.strip()) for p in pos_str.split(",") if p.strip()]
                index[current_term][docno] = positions
    return index

def all_doc_ids(index: Dict[str, Dict[str, List[int]]]) -> Set[str]:
    docs: Set[str] = set()
    for postings in index.values():
        docs.update(postings.keys())
    return docs

# ----------------------------
# Boolean / Phrase / Proximity search
# ----------------------------
PROX_RE = re.compile(r"#\s*(\d+)\s*\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)", re.IGNORECASE)

def preprocess_query_text(q: str, stopwords: Set[str], stemmer: PorterStemmer) -> List[str]:
    toks = tokenize(q)
    toks = [t for t in toks if t not in stopwords]
    return [stemmer.stem(t) for t in toks]

def eval_term(index, term: str) -> Set[str]:
    return set(index.get(term, {}).keys())

def eval_phrase(index, terms: List[str]) -> Set[str]:
    if not terms:
        return set()
    first_postings = index.get(terms[0], {})
    candidate_docs = set(first_postings.keys())
    for t in terms[1:]:
        candidate_docs &= set(index.get(t, {}).keys())
    if not candidate_docs:
        return set()
    matched_docs = set()
    for d in candidate_docs:
        pos_lists = [index[t][d] for t in terms]
        current = pos_lists[0]
        for i in range(1, len(pos_lists)):
            target = set(p+1 for p in current)
            current = [p for p in pos_lists[i] if p in target]
            if not current:
                break
        if current:
            matched_docs.add(d)
    return matched_docs

def eval_proximity(index, term1: str, term2: str, k: int) -> Set[str]:
    docs = set(index.get(term1, {}).keys()) & set(index.get(term2, {}).keys())
    matched = set()
    for d in docs:
        p1 = index[term1][d]
        p2 = index[term2][d]
        i = j = 0
        while i < len(p1) and j < len(p2):
            if abs(p1[i] - p2[j]) <= k:
                matched.add(d)
                break
            if p1[i] < p2[j]:
                i += 1
            else:
                j += 1
    return matched

def parse_boolean_query(q: str):
    m = PROX_RE.search(q)
    if m:
        k = int(m.group(1)); left = m.group(2); right = m.group(3)
        return ('PROX', k, left, right)

    if re.search(r"\bAND\b", q, flags=re.IGNORECASE):
        op = 'AND'
        parts = re.split(r"\bAND\b", q, flags=re.IGNORECASE)
    elif re.search(r"\bOR\b", q, flags=re.IGNORECASE):
        op = 'OR'
        parts = re.split(r"\bOR\b", q, flags=re.IGNORECASE)
    else:
        parts = [q]; op = None
    parts = [p.strip() for p in parts]

    if len(parts) == 1:
        if parts[0].startswith('"') and parts[0].endswith('"'):
            return ('PHRASE', parts[0][1:-1])
        if re.match(r"^\s*NOT\s+.+$", parts[0], flags=re.IGNORECASE):
            term_raw = re.sub(r"^\s*NOT\s+", "", parts[0], flags=re.IGNORECASE).strip()
            return ('NOT_SINGLE', term_raw)
        return ('SINGLE', parts[0])

    left_raw, right_raw = parts[0], parts[1]
    left_not = False; right_not = False
    if re.match(r"^NOT\s+", left_raw, flags=re.IGNORECASE):
        left_not = True
        left_raw = re.sub(r"^NOT\s+", "", left_raw, flags=re.IGNORECASE).strip()
    if re.match(r"^NOT\s+", right_raw, flags=re.IGNORECASE):
        right_not = True
        right_raw = re.sub(r"^NOT\s+", "", right_raw, flags=re.IGNORECASE).strip()
    return ('BIN', op, left_raw, right_raw, left_not, right_not)

def eval_boolean_query_struct(struct, index, stopwords, stemmer):
    universe = all_doc_ids(index)

    def to_docs(raw):
        raw = raw.strip()
        if raw.startswith('"') and raw.endswith('"'):
            terms = preprocess_query_text(raw[1:-1], stopwords, stemmer)
            return eval_phrase(index, terms)
        m = PROX_RE.fullmatch(raw)
        if m:
            k = int(m.group(1))
            t1 = preprocess_query_text(m.group(2), stopwords, stemmer)
            t2 = preprocess_query_text(m.group(3), stopwords, stemmer)
            if not t1 or not t2: return set()
            return eval_proximity(index, t1[0], t2[0], k)
        terms = preprocess_query_text(raw, stopwords, stemmer)
        if not terms: return set()
        docs = eval_term(index, terms[0])
        for t in terms[1:]:
            docs &= eval_term(index, t)
        return docs

    kind = struct[0]
    if kind == 'PROX':
        _k, left, right = struct[1], struct[2], struct[3]
        t1 = preprocess_query_text(left, stopwords, stemmer)
        t2 = preprocess_query_text(right, stopwords, stemmer)
        if not t1 or not t2: return set()
        return eval_proximity(index, t1[0], t2[0], _k)
    if kind == 'PHRASE':
        terms = preprocess_query_text(struct[1], stopwords, stemmer)
        return eval_phrase(index, terms)
    if kind == 'SINGLE':
        return to_docs(struct[1])
    if kind == 'NOT_SINGLE':
        return universe - to_docs(struct[1])
    if kind == 'BIN':
        _, op, left_raw, right_raw, left_not, right_not = struct
        left_docs = to_docs(left_raw)
        right_docs = to_docs(right_raw)
        if left_not: left_docs = universe - left_docs
        if right_not: right_docs = universe - right_docs
        return left_docs & right_docs if op == 'AND' else left_docs | right_docs
    return set()

# ----------------------------
# Ranked retrieval (TFIDF per lecture)
# ----------------------------
def ranked_scores(index, query_text: str, N_docs: int, stopwords: Set[str], stemmer: PorterStemmer):
    q_terms = preprocess_query_text(query_text, stopwords, stemmer)
    if not q_terms:
        return {}
    candidates = set()
    for t in q_terms:
        candidates |= set(index.get(t, {}).keys())
    idf = {}
    for t in q_terms:
        df = len(index.get(t, {}))
        idf[t] = math.log10(N_docs / df) if df > 0 else 0.0
    scores = defaultdict(float)
    for t in q_terms:
        postings = index.get(t, {})
        if not postings: continue
        w_idf = idf[t]
        if w_idf == 0.0: continue
        for d, positions in postings.items():
            tf = len(positions)
            w_td = (1.0 + (math.log10(tf) if tf > 0 else 0.0)) * w_idf
            scores[d] += w_td
    return scores

# ----------------------------
# CLI
# ----------------------------
def add_subcommands(ap):
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_pre = sub.add_parser("preprocess", help="Run preprocessing only.")
    ap_pre.add_argument("--trec_path", required=True)
    ap_pre.add_argument("--stopwords_path", required=True)
    ap_pre.add_argument("--out_path", required=True)

    ap_idx = sub.add_parser("index", help="Build positional inverted index.")
    ap_idx.add_argument("--preprocessed_jsonl", required=True)
    ap_idx.add_argument("--out_index", required=True)

    ap_bool = sub.add_parser("search_boolean", help="Execute boolean/phrase/proximity queries.")
    ap_bool.add_argument("--index_path", required=True)
    ap_bool.add_argument("--queries_path", required=True)
    ap_bool.add_argument("--stopwords_path", required=True)
    ap_bool.add_argument("--out_path", required=True)

    ap_rank = sub.add_parser("search_ranked", help="Execute TFIDF ranked retrieval (lecture formula).")
    ap_rank.add_argument("--index_path", required=True)
    ap_rank.add_argument("--queries_path", required=True)
    ap_rank.add_argument("--stopwords_path", required=True)
    ap_rank.add_argument("--out_path", required=True)

    return ap

def main():
    ap = argparse.ArgumentParser(description="IR pipeline: preprocess, index, boolean/phrase/proximity, TFIDF ranked.")
    ap = add_subcommands(ap)
    args = ap.parse_args()

    if args.cmd == "preprocess":
        stops = load_stopwords(args.stopwords_path)
        stemmer = PorterStemmer()
        n_docs = 0
        with open(args.out_path, "w", encoding="utf-8") as out:
            for doc in iter_trec_docs(args.trec_path):
                n_docs += 1
                processed = preprocess_doc(doc, stops, stemmer)
                out.write(json.dumps(processed, ensure_ascii=False) + "\n")
        print(f"Done. Processed {n_docs} docs -> {args.out_path}")
        return

    if args.cmd == "index":
        index = build_positional_index(args.preprocessed_jsonl)
        write_index_txt(index, args.out_index)
        print(f"Wrote positional inverted index to {args.out_index} (terms: {len(index)})")
        return

    if args.cmd == "search_boolean":
        index = load_index_from_txt(args.index_path)
        stops = load_stopwords(args.stopwords_path)
        stemmer = PorterStemmer()
        lines = []
        with open(args.queries_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                qnum, qraw = line.split(" ", 1)
                struct = parse_boolean_query(qraw.strip())
                docs = eval_boolean_query_struct(struct, index, stops, stemmer)
                def keyfn(x):
                    try: return (0, int(x))
                    except: return (1, x)
                for d in sorted(docs, key=keyfn):
                    lines.append(f"{qnum},{d}")
        with open(args.out_path, "w", encoding="utf-8") as out:
            out.write("\n".join(lines) + ("\n" if lines else ""))
        print(f"Wrote boolean results to {args.out_path} (lines: {len(lines)})")
        return

    if args.cmd == "search_ranked":
        index = load_index_from_txt(args.index_path)
        stops = load_stopwords(args.stopwords_path)
        stemmer = PorterStemmer()
        N = len(all_doc_ids(index))
        lines = []
        with open(args.queries_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                qnum, qraw = line.split(" ", 1)
                scores = ranked_scores(index, qraw, N, stops, stemmer)
                items = sorted(scores.items(), key=lambda kv: (-kv[1], int(kv[0]) if kv[0].isdigit() else kv[0]))[:150]
                for doc, sc in items:
                    lines.append(f"{qnum},{doc},{sc:.4f}")
        with open(args.out_path, "w", encoding="utf-8") as out:
            out.write("\n".join(lines) + ("\n" if lines else ""))
        print(f"Wrote ranked results to {args.out_path} (lines: {len(lines)})")
        return

if __name__ == "__main__":
    main()