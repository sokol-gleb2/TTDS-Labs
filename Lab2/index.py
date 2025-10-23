#!/usr/bin/env python3
"""
TTDS — Lab 2 (Indexing + Boolean, Phrase & Proximity Search)

Works directly on the **XML** version of the provided collections.

Features
- Tokenisation + case-folding
- Optional stopword removal and Porter stemming
- Positional inverted index: term -> {doc_id: [positions]}
- Boolean search: AND / OR / NOT (NOT has highest precedence; AND over OR)
- Phrase search: "quoted phrase"
- Proximity search: #k(term1, term2) — within k tokens (any order)
- Includes <HEADLINE> (then <TEXT>) when indexing TREC-style XML
- Saves a readable index dump to disk; can also save/load a compact JSON index

Usage examples
--------------
# 1) Build index from XML files contained in a zip (as provided for Lab 2):
#    python ttds_lab2_xml.py build --zip /path/to/collections-2.zip --xml-dir trec
#      (If unsure about the dir inside the zip, run with --list to inspect)
#
# 2) Build index from a directory on disk containing .xml files:
#    python ttds_lab2_xml.py build --xml-root /path/to/dir
#
# 3) Run the lab query file after building:
#    python ttds_lab2_xml.py query --queries-file queries.lab2.txt
#
# 4) Ad‑hoc queries (Boolean / Phrase / Proximity):
#    python ttds_lab2_xml.py query --q 'Scotland'
#    python ttds_lab2_xml.py query --q 'income OR taxes'
#    python ttds_lab2_xml.py query --q 'income AND NOT taxes'
#    python ttds_lab2_xml.py query --q '"income taxes"'
#    python ttds_lab2_xml.py query --q '#10(income, taxes)'
#    python ttds_lab2_xml.py query --q '"middle east" AND peace'
#
# 5) Toggle preprocessing:
#    --no-stops to disable stopword removal
#    --no-stem  to disable stemming
#    (Defaults: stopwords **on**, stemming **on**)
#
# Outputs
#  - data/index.json        : compact index (for fast reload)
#  - data/index.txt         : human-readable postings with positions
#  - data/doc_lengths.json  : doc lengths in tokens (post-preprocessing)
#  - data/docs.json         : doc metadata (doc_id -> {docno, headline_len})
"""
from __future__ import annotations
import argparse
import io
import json
import math
import os
import re
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Iterable, Tuple, Set, Optional

# -----------------------------
# Preprocessing
# -----------------------------
DEFAULT_STOPWORDS = {
    # A light English list; tweak/extend as needed
    'a','an','the','and','or','not','is','are','was','were','be','been','being',
    'of','to','in','on','for','with','as','by','at','from','that','this','these','those',
    'it','its','but','if','then','than','so','such','too','very','into','about','over','under',
    'no','nor','can','could','would','should','will','shall','may','might','must','do','does','did',
    'have','has','had','having','you','your','yours','he','she','they','them','his','her','their','we','our',
    'i','me','my','mine','him','hers','theirs','us','ours'
}

class OptionalPorter:
    """Minimal wrapper: uses NLTK's PorterStemmer if available; no-op otherwise."""
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._stemmer = None
        if enabled:
            try:
                from nltk.stem import PorterStemmer  # type: ignore
                self._stemmer = PorterStemmer()
            except Exception:
                self.enabled = False
    def stem(self, token: str) -> str:
        if not self.enabled or not self._stemmer:
            return token
        return self._stemmer.stem(token)

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())

# -----------------------------
# TREC XML parsing (without a global root)
# -----------------------------
DOC_START = re.compile(r"<DOC>\s*$")
DOC_END   = re.compile(r"</DOC>\s*$")
TAG = re.compile(r"<(/?)([A-Z]+)>\s*$")

@dataclass
class Document:
    docno: str
    headline: str
    text: str

    @property
    def as_text(self) -> str:
        # Concatenate HEADLINE then TEXT (per Lab/CW guidance)
        parts = []
        if self.headline:
            parts.append(self.headline.strip())
        if self.text:
            parts.append(self.text.strip())
        return "\n".join(parts)


def iter_trec_docs_from_stream(stream: Iterable[str]) -> Iterable[Document]:
    """Parse a TREC-ish XML stream into Document objects.
    Expects blocks: <DOC> ... <DOCNO>..</DOCNO> <HEADLINE>..</HEADLINE> <TEXT>..</TEXT> ... </DOC>
    HEADLINE may be empty/missing in some docs; handle robustly.
    """
    in_doc = False
    buf: List[str] = []
    for line in stream:
        if not in_doc and DOC_START.match(line):
            in_doc = True
            buf = []
            continue
        if in_doc:
            if DOC_END.match(line):
                raw = "".join(buf)
                # Extract fields with tolerant regex (DOTALL, optional tags)
                docno = re.search(r"<DOCNO>\s*(.*?)\s*</DOCNO>", raw, re.DOTALL)
                headline = re.search(r"<HEADLINE>\s*(.*?)\s*</HEADLINE>", raw, re.DOTALL)
                text = re.search(r"<TEXT>\s*(.*?)\s*</TEXT>", raw, re.DOTALL)
                yield Document(
                    docno=docno.group(1).strip() if docno else "",
                    headline=headline.group(1).strip() if headline else "",
                    text=text.group(1).strip() if text else "",
                )
                in_doc = False
                buf = []
            else:
                buf.append(line)


def iter_trec_docs_from_file(path: str) -> Iterable[Document]:
    with io.open(path, 'r', encoding='utf-8', errors='ignore') as f:
        yield from iter_trec_docs_from_stream(f)

# -----------------------------
# Indexing
# -----------------------------
@dataclass
class Index:
    postings: Dict[str, Dict[str, List[int]]]
    df: Dict[str, int]
    docs: Dict[str, Dict[str, int]]  # doc_id -> {"headline_len": int}
    doc_lengths: Dict[str, int]

    def dump_readable(self, path: str) -> None:
        with io.open(path, 'w', encoding='utf-8') as out:
            for term in sorted(self.postings.keys()):
                plist = self.postings[term]
                parts = []
                for doc_id in sorted(plist.keys()):
                    positions = ",".join(str(p) for p in plist[doc_id])
                    parts.append(f"{doc_id}:[{positions}]")
                df = self.df.get(term, len(plist))
                out.write(f"{term} ({df}): " + "; ".join(parts) + "\n")

    def save(self, base_dir: str) -> None:
        os.makedirs(base_dir, exist_ok=True)
        with open(os.path.join(base_dir, 'index.json'), 'w', encoding='utf-8') as f:
            json.dump(self.postings, f)
        with open(os.path.join(base_dir, 'doc_lengths.json'), 'w', encoding='utf-8') as f:
            json.dump(self.doc_lengths, f)
        with open(os.path.join(base_dir, 'docs.json'), 'w', encoding='utf-8') as f:
            json.dump(self.docs, f)
        with open(os.path.join(base_dir, 'df.json'), 'w', encoding='utf-8') as f:
            json.dump(self.df, f)

    @staticmethod
    def load(base_dir: str) -> 'Index':
        with open(os.path.join(base_dir, 'index.json'), 'r', encoding='utf-8') as f:
            postings = json.load(f)
        with open(os.path.join(base_dir, 'doc_lengths.json'), 'r', encoding='utf-8') as f:
            doc_lengths = json.load(f)
        with open(os.path.join(base_dir, 'docs.json'), 'r', encoding='utf-8') as f:
            docs = json.load(f)
        if os.path.exists(os.path.join(base_dir, 'df.json')):
            with open(os.path.join(base_dir, 'df.json'), 'r', encoding='utf-8') as f:
                df = json.load(f)
        else:
            df = {t: len(pl) for t, pl in postings.items()}
        # Convert keys in nested dicts to int lists if json restored as strings
        postings = {t: {d: list(map(int, pos)) for d, pos in pl.items()} for t, pl in postings.items()}
        doc_lengths = {d: int(n) for d, n in doc_lengths.items()}
        # docs dict values are small ints already
        return Index(postings=postings, df=df, docs=docs, doc_lengths=doc_lengths)


def build_index_from_docs(docs: Iterable[Document], *,
                          use_stops: bool=True, use_stem: bool=True,
                          stopwords: Optional[Set[str]]=None) -> Index:
    stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS
    stemmer = OptionalPorter(enabled=use_stem)

    postings: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    doc_lengths: Dict[str, int] = {}
    docs_meta: Dict[str, Dict[str, int]] = {}

    for doc in docs:
        doc_id = doc.docno or "DOC_{}".format(len(doc_lengths)+1)
        # Tokenise headline and text; compute headline length for ref (not used in positions by itself)
        headline_tokens = tokenize(doc.headline)
        text_tokens = tokenize(doc.text)
        tokens = headline_tokens + text_tokens
        # Apply stops + stemming
        processed: List[str] = []
        for tok in tokens:
            if use_stops and tok in stopwords:
                continue
            tok = stemmer.stem(tok) if use_stem else tok
            processed.append(tok)
        # Save headline length *after* preprocessing, so positions start at 0 and proceed through headline then text
        headline_len_after = 0
        if headline_tokens:
            processed_headline: List[str] = []
            for tok in headline_tokens:
                if use_stops and tok in stopwords:
                    continue
                tok = stemmer.stem(tok) if use_stem else tok
                processed_headline.append(tok)
            headline_len_after = len(processed_headline)
        # Index positions
        for pos, term in enumerate(processed):
            postings[term][doc_id].append(pos)
        doc_lengths[doc_id] = len(processed)
        docs_meta[doc_id] = {"headline_len": headline_len_after}

    df = {t: len(pl) for t, pl in postings.items()}
    return Index(postings=postings, df=df, docs=docs_meta, doc_lengths=doc_lengths)

# -----------------------------
# Query parsing & evaluation
# -----------------------------
# We support a tiny query language:
#  - tokens (terms)
#  - quoted phrases: "new york"
#  - proximity: #10(term1, term2)
#  - boolean: AND, OR, NOT (NOT > AND > OR)

Q_TOKEN = re.compile(r"\s+|(\()|(\))|(\#\d+\s*\(\s*[^,\s]+\s*,\s*[^\)\s]+\s*\))|(\"[^\"]+\")|(AND|OR|NOT)", re.IGNORECASE | re.VERBOSE)

@dataclass
class Node:
    kind: str  # TERM | PHRASE | PROX | AND | OR | NOT
    value: Optional[str] = None
    children: Tuple['Node', ...] = ()


def lex(query: str) -> List[str]:
    tokens: List[str] = []
    i = 0
    while i < len(query):
        m = Q_TOKEN.match(query, i)
        if not m:
            # read a bare term until whitespace or operator
            j = i
            while j < len(query) and not Q_TOKEN.match(query, j):
                j += 1
            tokens.append(query[i:j])
            i = j
        else:
            if m.group(0).strip():
                tokens.append(m.group(0))
            i = m.end()
    return tokens


def parse(query: str) -> Node:
    # Shunting-yard to AST with precedence NOT > AND > OR. Phrases/PROX become atomic terms.
    prec = {"NOT": 3, "AND": 2, "OR": 1}
    output: List[Node] = []
    ops: List[str] = []

    def push_term(tok: str) -> None:
        if tok.startswith('"') and tok.endswith('"'):
            output.append(Node('PHRASE', tok[1:-1]))
        elif tok.upper().startswith('#'):
            # #k(term1, term2)
            m = re.match(r"#(\d+)\s*\(\s*([^,\s]+)\s*,\s*([^\)\s]+)\s*\)", tok, re.IGNORECASE)
            if not m:
                raise ValueError(f"Bad proximity: {tok}")
            k, a, b = int(m.group(1)), m.group(2), m.group(3)
            output.append(Node('PROX', json.dumps({"k": k, "a": a, "b": b})))
        else:
            output.append(Node('TERM', tok))

    toks = lex(query)
    i = 0
    while i < len(toks):
        t = toks[i]
        tu = t.upper()
        if t == '(':
            ops.append(t)
        elif t == ')':
            while ops and ops[-1] != '(':
                op = ops.pop()
                if op == 'NOT':
                    b = output.pop()
                    output.append(Node('NOT', children=(b,)))
                else:
                    b = output.pop(); a = output.pop()
                    output.append(Node(op, children=(a,b)))
            if not ops:
                raise ValueError("Mismatched parentheses")
            ops.pop()
        elif tu in ('AND','OR','NOT'):
            while ops and ops[-1] != '(' and prec.get(ops[-1],0) >= prec[tu]:
                op = ops.pop()
                if op == 'NOT':
                    b = output.pop()
                    output.append(Node('NOT', children=(b,)))
                else:
                    b = output.pop(); a = output.pop()
                    output.append(Node(op, children=(a,b)))
            ops.append(tu)
        else:
            push_term(t)
        i += 1
    while ops:
        op = ops.pop()
        if op == '(':
            raise ValueError("Mismatched parentheses")
        if op == 'NOT':
            b = output.pop()
            output.append(Node('NOT', children=(b,)))
        else:
            b = output.pop(); a = output.pop()
            output.append(Node(op, children=(a,b)))
    if len(output) != 1:
        raise ValueError("Parse error")
    return output[0]

# -----------------------------
# Evaluation helpers
# -----------------------------

def preprocess_terms(terms: Iterable[str], *, use_stops: bool, use_stem: bool) -> List[str]:
    stemmer = OptionalPorter(enabled=use_stem)
    out: List[str] = []
    for t in terms:
        toks = tokenize(t)
        for tok in toks:
            if use_stops and tok in DEFAULT_STOPWORDS:
                continue
            out.append(stemmer.stem(tok) if use_stem else tok)
    return out


def postings_for_term(index: Index, term: str) -> Dict[str, List[int]]:
    return index.postings.get(term, {})


def boolean_and(a: Dict[str, List[int]], b: Dict[str, List[int]]) -> Dict[str, List[int]]:
    docs = set(a.keys()) & set(b.keys())
    return {d: [] for d in docs}


def boolean_or(a: Dict[str, List[int]], b: Dict[str, List[int]]) -> Dict[str, List[int]]:
    docs = set(a.keys()) | set(b.keys())
    return {d: [] for d in docs}


def boolean_not(universe: Set[str], a: Dict[str, List[int]]) -> Dict[str, List[int]]:
    docs = universe - set(a.keys())
    return {d: [] for d in docs}


def phrase_match(index: Index, terms: List[str]) -> Dict[str, List[int]]:
    if not terms:
        return {}
    lists = [index.postings.get(t, {}) for t in terms]
    common_docs = set.intersection(*(set(d.keys()) for d in lists)) if lists else set()
    result: Dict[str, List[int]] = {}
    for d in common_docs:
        poss = [lists[i][d] for i in range(len(terms))]
        # For each position of the first term, check chain +1 increments
        hits: List[int] = []
        first_positions = poss[0]
        others = poss[1:]
        others_sets = [set(p) for p in others]
        for p in first_positions:
            ok = True
            cur = p
            for k in range(len(others)):
                cur += 1
                if cur not in others_sets[k]:
                    ok = False; break
            if ok:
                hits.append(p)
        if hits:
            result[d] = hits
    return result


def proximity_match(index: Index, a: str, b: str, k: int) -> Dict[str, List[int]]:
    A = index.postings.get(a, {})
    B = index.postings.get(b, {})
    docs = set(A.keys()) & set(B.keys())
    res: Dict[str, List[int]] = {}
    for d in docs:
        pa = A[d]; pb = B[d]
        i=j=0
        hits: List[int] = []
        pa_sorted = sorted(pa); pb_sorted = sorted(pb)
        # two-pointer sweep for |pa - pb| <= k (any order)
        while i < len(pa_sorted) and j < len(pb_sorted):
            if abs(pa_sorted[i]-pb_sorted[j]) <= k:
                hits.append(min(pa_sorted[i], pb_sorted[j]))
                # advance the smaller pointer
                if pa_sorted[i] <= pb_sorted[j]:
                    i += 1
                else:
                    j += 1
            elif pa_sorted[i] < pb_sorted[j]:
                i += 1
            else:
                j += 1
        if hits:
            res[d] = hits
    return res


def eval_ast(node: Node, index: Index, *, use_stops: bool, use_stem: bool) -> Dict[str, List[int]]:
    universe: Set[str] = set(index.doc_lengths.keys())
    if node.kind == 'TERM':
        terms = preprocess_terms([node.value or ''], use_stops=use_stops, use_stem=use_stem)
        # Single term after preprocessing may map to 0 or 1 tokens; if >1, OR them.
        result: Dict[str, List[int]] = {}
        for t in terms:
            p = postings_for_term(index, t)
            result = boolean_or(result, p) if result else p
        return result
    if node.kind == 'PHRASE':
        terms = preprocess_terms(node.value.split(), use_stops=use_stops, use_stem=use_stem)
        return phrase_match(index, terms)
    if node.kind == 'PROX':
        spec = json.loads(node.value or '{}')
        a_raw, b_raw, k = spec['a'], spec['b'], int(spec['k'])
        a = preprocess_terms([a_raw], use_stops=use_stops, use_stem=use_stem)
        b = preprocess_terms([b_raw], use_stops=use_stops, use_stem=use_stem)
        if not a or not b:
            return {}
        return proximity_match(index, a[0], b[0], k)
    if node.kind == 'NOT':
        child = eval_ast(node.children[0], index, use_stops=use_stops, use_stem=use_stem)
        return boolean_not(universe, child)
    if node.kind in ('AND','OR'):
        left = eval_ast(node.children[0], index, use_stops=use_stops, use_stem=use_stem)
        right= eval_ast(node.children[1], index, use_stops=use_stops, use_stem=use_stem)
        return boolean_and(left,right) if node.kind=='AND' else boolean_or(left,right)
    raise ValueError(f"Unknown node kind: {node.kind}")

# -----------------------------
# I/O helpers for sources
# -----------------------------

def list_zip(xml_zip: str) -> None:
    with zipfile.ZipFile(xml_zip, 'r') as z:
        for n in z.namelist():
            print(n)


def iter_xml_files_from_zip(xml_zip: str, inner_dir: Optional[str]) -> Iterable[Tuple[str, io.TextIOBase]]:
    z = zipfile.ZipFile(xml_zip, 'r')
    for name in z.namelist():
        if inner_dir and not name.startswith(inner_dir.rstrip('/') + '/'):
            continue
        if name.lower().endswith('.xml'):
            data = z.read(name).decode('utf-8', errors='ignore')
            yield name, io.StringIO(data)


def iter_docs_from_sources(args) -> Iterable[Document]:
    if args.zip:
        for name, stream in iter_xml_files_from_zip(args.zip, args.xml_dir):
            yield from iter_trec_docs_from_stream(stream)
    elif args.xml_root:
        for root, _, files in os.walk(args.xml_root):
            for fn in files:
                if fn.lower().endswith('.xml'):
                    path = os.path.join(root, fn)
                    yield from iter_trec_docs_from_file(path)
    else:
        raise SystemExit("Provide --zip or --xml-root")

# -----------------------------
# CLI
# -----------------------------

def cmd_build(args):
    if args.list:
        list_zip(args.zip)
        return
    docs = list(iter_docs_from_sources(args))
    index = build_index_from_docs(
        docs,
        use_stops=not args.no_stops,
        use_stem=not args.no_stem,
    )
    os.makedirs(args.out, exist_ok=True)
    index.dump_readable(os.path.join(args.out, 'index.txt'))
    index.save(args.out)
    print(f"Indexed {len(index.doc_lengths)} documents; vocabulary={len(index.postings)}")
    print(f"Saved to {args.out}/index.json, {args.out}/index.txt")


def cmd_query(args):
    index = Index.load(args.index)
    use_stops=not args.no_stops
    use_stem=not args.no_stem

    def run_one(q: str) -> Tuple[str, List[str]]:
        ast = parse(q)
        match = eval_ast(ast, index, use_stops=use_stops, use_stem=use_stem)
        docs = sorted(match.keys())
        return q, docs

    pairs: List[Tuple[str, List[str]]] = []
    if args.queries_file:
        with open(args.queries_file, 'r', encoding='utf-8') as f:
            line = f.read().strip()
            # Format per Lab file: q1: Scotland q2: ...
            # Split at q#: tokens, keep the query text following each label up to next label
            m = re.findall(r"q\d+:\s*([^q]+)", line)
            # Fallback: split on q\d: boundaries
            if not m:
                pieces = re.split(r"q\d+:\s*", line)[1:]
            else:
                pieces = [s.strip() for s in m]
            for q in pieces:
                if q:
                    pairs.append(run_one(q))
    elif args.q:
        pairs.append(run_one(args.q))
    else:
        raise SystemExit("Provide --q or --queries-file")

    for q, docs in pairs:
        print(f"Q: {q}")
        print(f"Docs ({len(docs)}): {', '.join(docs)}\n")


def main():
    p = argparse.ArgumentParser(description="TTDS Lab 2 — XML index & search")
    sub = p.add_subparsers()

    b = sub.add_parser('build', help='Build the index from XML sources')
    b.add_argument('--zip', type=str, help='Path to collections-2.zip')
    b.add_argument('--xml-dir', type=str, help='Directory inside the zip that has XML files (e.g., trec)')
    b.add_argument('--xml-root', type=str, help='Directory on disk with .xml files')
    b.add_argument('--out', type=str, default='data', help='Output dir for index files')
    b.add_argument('--list', action='store_true', help='List zip contents and exit')
    b.add_argument('--no-stops', action='store_true', help='Disable stopword removal')
    b.add_argument('--no-stem', action='store_true', help='Disable stemming')
    b.set_defaults(func=cmd_build)

    q = sub.add_parser('query', help='Run queries over a built index')
    q.add_argument('--index', type=str, default='data', help='Directory containing index.json')
    q.add_argument('--q', type=str, help='Single query string')
    q.add_argument('--queries-file', type=str, help='Run all queries from Lab 2 file')
    q.add_argument('--no-stops', action='store_true', help='Disable stopword removal at query time')
    q.add_argument('--no-stem', action='store_true', help='Disable stemming at query time')
    q.set_defaults(func=cmd_query)

    args = p.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        p.print_help()

if __name__ == '__main__':
    main()