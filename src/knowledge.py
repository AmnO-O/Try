#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KIM-style lexical knowledge for the xattn comment/video pair.

This module builds the five WordNet relation features used by KIM (Chen et al.,
2018) between every comment word and every video word:

    0 synonymy
    1 antonymy
    2 hypernymy   (the video word is an ancestor of the comment word)
    3 hyponymy    (the video word is a descendant of the comment word)
    4 co-hyponymy (same direct hypernym, different synset)

The direction follows KIM's examples: for ``(dog, canid)`` the hypernymy
feature is ``1 - 1/8 = .875`` and hyponymy is zero; for ``(canid, dog)`` the
two are swapped.  The output is a dense ``[n_comment_words, n_video_words, 5]``
float32 matrix.  Padded/subword positions are handled by the caller, not here.

The module is deliberately optional.  ``train_xattn.py --kim-local`` imports
it; every other path keeps the original five-tuple data pipeline untouched.
WordNet itself is English.  Italian and Dutch lemmas are looked up through
NLTK's Open Multilingual WordNet (``lang='ita'`` / ``lang='nld'``); the synset
relations are language-independent, so a relation between two Italian lemmas
is found by intersecting the synsets that contain those lemmas.  If OMW is not
installed, non-English rows get zero knowledge rather than crashing the run.

Install the data once with::

    python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-2.0')"

The boundaries are intentionally lexical.  This is KIM's knowledge source, not
a general stereotype/commonsense knowledge base.
"""

from __future__ import annotations

import os
import pickle
import re
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

FEATURE_NAMES: Tuple[str, ...] = (
    'synonymy',
    'antonymy',
    'hypernymy',
    'hyponymy',
    'co_hyponymy',
)
KNOWLEDGE_DIM = len(FEATURE_NAMES)

# NLTK's WordNet language codes.  ``None`` means the default English WordNet.
LANG_TO_WN: Dict[str, Optional[str]] = {
    'EN': None,
    'IT': 'ita',
    'NL': 'nld',
}

# A conservative multilingual word pattern.  It also strips punctuation around
# hashtags/mentions, but keeps apostrophes and hyphens inside words.
_WORD_RE = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’-][0-9A-Za-zÀ-ÖØ-öø-ÿ]+)*")

# Very common function words.  They are not in WordNet as content lemmas and
# only make the pair loop bigger.  The list is deliberately short: removing a
# content word is worse than doing a few extra dictionary misses.
_STOPWORDS = frozenset(
    'a an the and or but if of to in on for with without at by from as is are '
    'was were be been being this that these those it its he she they them his '
    'her their you your i we our not no yes do does did can could should would '
    'will just so than too very'.split()
)


def _wordnet():
    """Import NLTK lazily so the original xattn paths do not need NLTK."""
    try:
        from nltk.corpus import wordnet as wn
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            'KIM knowledge needs nltk. Install it with `pip install nltk`, then '
            "run `python -c \"import nltk; nltk.download('wordnet'); "
            "nltk.download('omw-2.0')\"`."
        ) from exc
    return wn


@lru_cache(maxsize=200_000)
def _synsets(lang: str, word: str) -> Tuple:
    """Cached WordNet synsets for one surface word in one language."""
    wn = _wordnet()
    code = LANG_TO_WN.get(lang.upper())
    try:
        if code is None:
            return tuple(wn.synsets(word))
        return tuple(wn.synsets(word, lang=code))
    except LookupError as exc:  # missing WordNet / OMW data
        raise RuntimeError(
            'NLTK WordNet or OMW data is missing. Run '
            "`python -c \"import nltk; nltk.download('wordnet'); "
            "nltk.download('omw-2.0')\"` once, then retry."
        ) from exc


@lru_cache(maxsize=200_000)
def _ancestors(synset_name: str, max_edges: int) -> Tuple[Tuple[str, int], ...]:
    """All hypernym ancestors of a synset, with the shortest edge distance."""
    wn = _wordnet()
    start = wn.synset(synset_name)
    seen: Dict[str, int] = {synset_name: 0}
    frontier: List[Tuple[object, int]] = [(start, 0)]
    while frontier:
        node, dist = frontier.pop()
        if dist >= max_edges:
            continue
        for parent in node.hypernyms():
            name = parent.name()
            if name not in seen or dist + 1 < seen[name]:
                seen[name] = dist + 1
                frontier.append((parent, dist + 1))
    return tuple(sorted(seen.items()))


@lru_cache(maxsize=200_000)
def _direct_hypernyms(synset_name: str) -> Tuple[str, ...]:
    wn = _wordnet()
    return tuple(sorted(h.name() for h in wn.synset(synset_name).hypernyms()))


@lru_cache(maxsize=200_000)
def _antonym_synsets(synset_name: str) -> Tuple[str, ...]:
    """Antonym synsets, derived from the synset's English lemmas.

    OMW lemmas do not carry their own antonym pointers in NLTK, but the synset
    identity is shared across languages, so English antonym pointers are the
    correct language-independent relation.
    """
    wn = _wordnet()
    synset = wn.synset(synset_name)
    out = set()
    for lemma in synset.lemmas():
        for antonym in lemma.antonyms():
            out.add(antonym.synset().name())
    return tuple(sorted(out))


@lru_cache(maxsize=500_000)
def relation_vector(lang: str, word_a: str, word_b: str,
                    max_hypernym_edges: int = 8) -> Tuple[float, float, float, float, float]:
    """KIM's five relation features for one ordered word pair.

    ``word_a`` is on the comment side and ``word_b`` on the video side.
    """
    sa = _synsets(lang.upper(), word_a)
    sb = _synsets(lang.upper(), word_b)
    if not sa or not sb:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    names_a = {s.name() for s in sa}
    names_b = {s.name() for s in sb}

    synonymy = 1.0 if names_a & names_b else 0.0

    antonymy = 0.0
    for name in names_a:
        if set(_antonym_synsets(name)) & names_b:
            antonymy = 1.0
            break

    # Union the ancestor maps of all comment synsets, keeping the shortest
    # distance to each ancestor.  Then check which video synsets are ancestors.
    ancestors_a: Dict[str, int] = {}
    for s in sa:
        for name, dist in _ancestors(s.name(), max_hypernym_edges):
            if name not in ancestors_a or dist < ancestors_a[name]:
                ancestors_a[name] = dist
    hypernymy = 0.0
    for s in sb:
        dist = ancestors_a.get(s.name())
        if dist is not None and dist > 0:
            hypernymy = max(hypernymy, 1.0 - dist / float(max_hypernym_edges))

    ancestors_b: Dict[str, int] = {}
    for s in sb:
        for name, dist in _ancestors(s.name(), max_hypernym_edges):
            if name not in ancestors_b or dist < ancestors_b[name]:
                ancestors_b[name] = dist
    hyponymy = 0.0
    for s in sa:
        dist = ancestors_b.get(s.name())
        if dist is not None and dist > 0:
            hyponymy = max(hyponymy, 1.0 - dist / float(max_hypernym_edges))

    co_hyponymy = 0.0
    if synonymy == 0.0:
        for s in sa:
            h1 = set(_direct_hypernyms(s.name()))
            if not h1:
                continue
            for t in sb:
                if h1 & set(_direct_hypernyms(t.name())):
                    co_hyponymy = 1.0
                    break
            if co_hyponymy:
                break

    return (synonymy, antonymy, hypernymy, hyponymy, co_hyponymy)


def word_spans(text: str) -> List[Tuple[str, int, int]]:
    """Return ``(lowercased_word, start, end)`` for content-ish words."""
    out: List[Tuple[str, int, int]] = []
    for match in _WORD_RE.finditer(text):
        word = match.group(0).lower()
        if len(word) < 2 or word in _STOPWORDS:
            continue
        out.append((word, match.start(), match.end()))
    return out


def word_list(text: str) -> List[str]:
    return [w for w, _, _ in word_spans(text)]


def word_relation_matrix(comment_words: Sequence[str],
                         video_words: Sequence[str],
                         lang: str,
                         max_hypernym_edges: int = 8) -> np.ndarray:
    """Dense ``[C, V, 5]`` relation matrix for one word sequence pair.

    Duplicate words are collapsed before the pair loop; WordNet results are
    cached, so a repeated word in a comment costs a dict lookup, not another
    graph traversal.
    """
    c_unique = list(dict.fromkeys(comment_words))
    v_unique = list(dict.fromkeys(video_words))
    if not c_unique or not v_unique:
        return np.zeros((len(comment_words), len(video_words), KNOWLEDGE_DIM),
                        dtype=np.float32)

    unique = np.zeros((len(c_unique), len(v_unique), KNOWLEDGE_DIM), dtype=np.float32)
    for i, a in enumerate(c_unique):
        for j, b in enumerate(v_unique):
            unique[i, j] = relation_vector(lang.upper(), a, b, max_hypernym_edges)

    c_idx = {w: i for i, w in enumerate(c_unique)}
    v_idx = {w: i for i, w in enumerate(v_unique)}
    c_take = np.asarray([c_idx[w] for w in comment_words], dtype=np.int64)
    v_take = np.asarray([v_idx[w] for w in video_words], dtype=np.int64)
    return unique[c_take][:, v_take]


def token_word_indices(text: str,
                       offset_mapping: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Map each tokenizer token to a word index, or ``-1`` for special tokens.

    ``offset_mapping`` is the fast tokenizer's character span for each token.
    A subword token whose span falls inside a word gets that word's index; the
    ``[CLS]``/``[SEP]``/punctuation tokens get ``-1``.
    """
    spans = word_spans(text)
    starts = np.asarray([s for _, s, _ in spans], dtype=np.int64)
    ends = np.asarray([e for _, _, e in spans], dtype=np.int64)
    out = np.full(len(offset_mapping), -1, dtype=np.int64)
    if not spans:
        return out
    for tok_idx, (start, end) in enumerate(offset_mapping):
        if start == end:
            continue
        pos = int(np.searchsorted(starts, start, side='right') - 1)
        if pos >= 0 and start < ends[pos] and end > starts[pos]:
            out[tok_idx] = pos
    return out


class LexicalKnowledge:
    """WordNet-backed relation builder for one or more languages."""

    def __init__(self, max_hypernym_edges: int = 8):
        if max_hypernym_edges < 1:
            raise ValueError('max_hypernym_edges must be >= 1')
        self.max_hypernym_edges = int(max_hypernym_edges)

    def matrix(self, comment_text: str, video_text: str, lang: str) -> np.ndarray:
        cw = word_list(comment_text)
        vw = word_list(video_text)
        return word_relation_matrix(cw, vw, lang, self.max_hypernym_edges)


class KnowledgeCache:
    """Disk-backed cache from (language, comment text, video text) to matrices.

    The cache stores the word lists alongside the word-level matrix so the
    dataset can expand the matrix to subword tokens with the tokenizer's offset
    mapping.  It is deliberately row-text keyed: the same video/comment pair
    always gets the same matrix, regardless of split or seed.
    """

    VERSION = 1

    def __init__(self, path: Optional[str] = None, max_hypernym_edges: int = 8,
                 verbose: bool = True):
        self.path = path
        self.verbose = verbose
        self.knowledge = LexicalKnowledge(max_hypernym_edges=max_hypernym_edges)
        self.max_hypernym_edges = int(max_hypernym_edges)
        self._entries: Dict[Tuple[str, str, str], Tuple[List[str], List[str], np.ndarray]] = {}
        self._new = 0
        if path and os.path.exists(path):
            self.load()

    def _key(self, lang: str, comment_text: str, video_text: str):
        return (lang.upper(), comment_text, video_text)

    def get(self, lang: str, comment_text: str, video_text: str):
        key = self._key(lang, comment_text, video_text)
        hit = self._entries.get(key)
        if hit is not None:
            return hit
        cw = word_list(comment_text)
        vw = word_list(video_text)
        matrix = word_relation_matrix(cw, vw, lang, self.max_hypernym_edges)
        entry = (cw, vw, matrix)
        self._entries[key] = entry
        self._new += 1
        if self.verbose and self._new % 250 == 0:
            print(f'  [kim] cached {len(self._entries)} comment/video pairs', flush=True)
        return entry

    def load(self):
        if not self.path:
            return
        try:
            with open(self.path, 'rb') as fh:
                payload = pickle.load(fh)
            if payload.get('version') != self.VERSION:
                raise ValueError('cache version mismatch')
            if payload.get('max_hypernym_edges') != self.max_hypernym_edges:
                raise ValueError('cache max_hypernym_edges mismatch')
            self._entries.update(payload['entries'])
            if self.verbose:
                print(f'  [kim] loaded {len(self._entries)} cached pairs from {self.path}',
                      flush=True)
        except Exception as exc:  # noqa: BLE001
            # A stale/incompatible cache is not data loss: rebuild it in place.
            if self.verbose:
                print(f'  [kim] ignoring cache {self.path}: {type(exc).__name__}: {exc}',
                      flush=True)
            self._entries.clear()

    def save(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + '.tmp'
        with open(tmp, 'wb') as fh:
            pickle.dump({
                'version': self.VERSION,
                'max_hypernym_edges': self.max_hypernym_edges,
                'feature_names': FEATURE_NAMES,
                'entries': self._entries,
            }, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.path)
        if self.verbose:
            print(f'  [kim] saved {len(self._entries)} cached pairs to {self.path}',
                  flush=True)

    def __len__(self):
        return len(self._entries)


__all__ = [
    'FEATURE_NAMES',
    'KNOWLEDGE_DIM',
    'KnowledgeCache',
    'LexicalKnowledge',
    'relation_vector',
    'token_word_indices',
    'word_list',
    'word_relation_matrix',
    'word_spans',
]
