"""BM25 pool: sparse text retrieval over track metadata.

Default config: `music_only` query mode + `noun_tags` doc field + spaCy
content-token filter.

query_mode:
  - "history"    : full chat_history + cur_query
  - "music_only" : music-turn metadata from chat_history + cur_query only
                   (user/assistant natural-language utterances excluded as noise)

corpus_types: list of track-doc index fields. Two common sets:
  - DEFAULT_CORPUS_TYPES (4-field)
  - DEFAULT_CORPUS_TYPES + ["noun_tags"] (5-field, content-token TF boost)

`spacy_filter=True`: natural-language fragments (cur_query, user/assistant
history content) are reduced to content tokens (NOUN/PROPN/ADJ/VERB, alpha,
non-stop) via spaCy. Track metadata blocks and noun_tags are already
keyword-form, so the filter is not applied to them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import bm25s
from datasets import load_dataset
from loguru import logger

from mymodule.strategies.pool.base import BasePool
from mymodule.utils.seen import extract_session_seen_tracks, take_unseen_pairs

# Lazy-loaded process singleton spaCy model — only initialized when a
# BM25Pool with spacy_filter=True (or noun_tags-bearing corpus_types) is
# constructed. Avoids the import + model load cost when no caller asks.
_SPACY_NLP: Any = None
# POS tags considered "content" — proper nouns (artist/track names),
# nouns (genres, instruments), adjectives (mood/style), verbs (action).
_SPACY_CONTENT_POS = {"NOUN", "PROPN", "ADJ", "VERB"}


def _get_spacy_nlp() -> Any:
    """Lazy-load `en_core_web_sm` once per process. Disable parser/lemmatizer
    for speed — we only need POS tags + stopword/alpha checks."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy

        _SPACY_NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        logger.info("BM25Pool: loaded spaCy en_core_web_sm (parser/lemmatizer disabled)")
    return _SPACY_NLP


def _spacy_content_tokens(text: str) -> str:
    """Strip stopwords / punctuation / function words from `text` via spaCy,
    keeping only content tokens (NOUN/PROPN/ADJ/VERB, alpha, non-stop).

    Returns space-joined token surface forms. Empty string if input empty
    or no content tokens survive.
    """
    if not text or not text.strip():
        return ""
    doc = _get_spacy_nlp()(text)
    keep = [t.text for t in doc if t.pos_ in _SPACY_CONTENT_POS and t.is_alpha and not t.is_stop]
    return " ".join(keep)


_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "feature" / ".bm25_cache"

# Default 4-field corpus matching baselines/mcrs/retrieval_modules/bm25.py.
DEFAULT_CORPUS_TYPES = ["track_name", "artist_name", "album_name", "release_date"]

# 5th virtual doc field — not from HF dataset directly. Computed at index
# build time by feeding `track_name + artist_name + album_name` through
# spaCy and keeping only content tokens. Adding this alongside the
# original four fields effectively boosts content-token TF (each appears
# twice — once in natural form, once cleaned) without losing release_date
# numbers or label tokens. Default-on combination uses this.
NOUN_TAGS_FIELD = "noun_tags"
NOUN_TAGS_SOURCE_FIELDS = ["track_name", "artist_name", "album_name"]
DEFAULT_NOUN_TAGS_CORPUS = DEFAULT_CORPUS_TYPES + [NOUN_TAGS_FIELD]

# Crawl-sourced doc fields — populated from lyrics.jsonl (talkpl + LRCLIB +
# MusicBrainz). Each maps to a key in the JSONL record. Used as additional
# corpus_types when crawl enrichment is enabled.
CRAWL_FIELDS: dict[str, str] = {
    "lyrics_crawl": "lyrics",
    "caption_crawl": "caption",
    "mb_tags": "mb_tags",
    "label": "label",
    "country": "country",
    "mb_release_date": "mb_release_date",
    "tempo": "tempo",
    "key": "key",
    "chord": "chord",
}
# All crawl fields combined — convenient shorthand for full enrichment.
CRAWL_ALL_FIELDS = list(CRAWL_FIELDS.keys())

# Virtual tag fields from HF dataset `tag_list` column.
TAG_LIST_RAW_FIELD = "tag_list_raw"
TAG_LIST_CLEAN_FIELD = "tag_list_clean"
# Tags to drop — user-generated noise / non-music metadata from Last.fm/Pandora.
_TAG_NOISE_PREFIXES = ("via ", "ion b ", "3 of ", "4 of ", "5 of ")


def _clean_tag_list(tags: list[str]) -> list[str]:
    """Deduplicate + filter tag_list for BM25 indexing.

    - Lowercase + deduplicate (case-insensitive)
    - Remove tags containing digits (e.g. "3 of 10 stars")
    - Remove tags shorter than 2 chars
    - Remove known noise prefixes ("via pandora", "ion b chill station")
    """
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        t = tag.strip().lower()
        if not t or len(t) < 2:
            continue
        if any(c.isdigit() for c in t):
            continue
        if any(t.startswith(p) for p in _TAG_NOISE_PREFIXES):
            continue
        if t in seen:
            continue
        seen.add(t)
        result.append(t)
    return result


def _load_crawl_from_kv() -> dict[str, dict]:
    """Load crawled metadata records from RocksDB (`track:crawl:*` prefix).

    Returns `{track_id: record}` dict. Raises `FileNotFoundError` with an actionable
    message if the KV store isn't built or has no crawl entries — pointing the
    caller at `--build-crawl`. Use this in place of reading the JSONL directly
    so all consumers share a single source.
    """
    from mymodule.feature import kvdb as kvdb_mod
    from mymodule.feature.kvdb import KVStore

    try:
        kv = KVStore.open(kvdb_mod.DB_PATH)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "KVStore not built. Run: uv run python -m mymodule.feature.kvdb --build "
            "and then uv run python -m mymodule.feature.kvdb --build-crawl"
        ) from e
    if kv.crawl_count() is None:
        raise FileNotFoundError("Crawl data not in KV. Run: uv run python -m mymodule.feature.kvdb --build-crawl")
    return dict(kv.iter_track_crawl())


class BM25Pool(BasePool):
    """BM25 sparse retrieval pool. Default = music_only + noun_tags + spaCy."""

    def __init__(
        self,
        n_candidates: int = 100,
        query_mode: str = "music_only",
        corpus_types: list[str] | None = None,
        spacy_filter: bool = True,
        **kwargs,
    ) -> None:
        if query_mode not in ("history", "music_only"):
            raise ValueError(
                f"query_mode must be 'history' or 'music_only', got {query_mode!r}."
            )
        self.n_candidates = n_candidates
        self.query_mode = query_mode
        self.corpus_types = corpus_types or DEFAULT_NOUN_TAGS_CORPUS
        self.spacy_filter = spacy_filter
        self._needs_crawl = any(ct in CRAWL_FIELDS for ct in self.corpus_types)
        self._metadata_dict: dict[str, str] | None = None
        self._build_or_load_index()
        if spacy_filter:
            _get_spacy_nlp()  # eager load so first inference call doesn't pay startup
        crawl_info = ", crawl=kv" if self._needs_crawl else ""
        logger.info(
            f"BM25Pool: {len(self.track_ids)} tracks, query_mode={query_mode}, "
            f"corpus_types={self.corpus_types}, spacy_filter={spacy_filter}, "
            f"n_candidates={n_candidates}{crawl_info}"
        )

    def _build_or_load_index(self) -> None:
        index_dir = _CACHE_DIR / "_".join(self.corpus_types)
        ids_path = index_dir / "track_ids.json"
        meta_path = index_dir / "metadata_strings.json"

        if index_dir.exists() and ids_path.exists() and meta_path.exists():
            self.bm25 = bm25s.BM25.load(str(index_dir), load_corpus=False)
            with open(ids_path) as f:
                self.track_ids = json.load(f)
            with open(meta_path) as f:
                self._metadata_dict = json.load(f)
            return

        ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
        self.track_ids = list(ds["track_id"])

        # Load crawl records from KV if any crawl field is requested.
        crawl_data: dict[str, dict] = {}
        if self._needs_crawl:
            crawl_data = _load_crawl_from_kv()
            logger.info(f"BM25Pool: loaded {len(crawl_data)} crawl records from KV (track:crawl:*)")

        # If the user requested the virtual `noun_tags` field, eagerly load
        # spaCy so we can compute it for every track during the build below.
        needs_noun_tags = NOUN_TAGS_FIELD in self.corpus_types
        if needs_noun_tags:
            _get_spacy_nlp()
            logger.info(f"BM25Pool: building noun_tags for {len(self.track_ids)} tracks (spaCy)…")

        corpus = []
        metadata_dict = {}
        for row in ds:
            tid = row["track_id"]
            crawl_rec = crawl_data.get(tid, {})
            parts = []
            for field in self.corpus_types:
                if field == NOUN_TAGS_FIELD:
                    # Virtual field: spaCy content-token extraction over
                    # track_name + artist_name + album_name.
                    src = []
                    for sf in NOUN_TAGS_SOURCE_FIELDS:
                        v = row.get(sf)
                        if v is None:
                            continue
                        if isinstance(v, list):
                            v = " ".join(str(x) for x in v)
                        src.append(str(v))
                    tags = _spacy_content_tokens(" ".join(src))
                    if tags:
                        parts.append(f"{NOUN_TAGS_FIELD}: {tags}")
                    continue
                if field == TAG_LIST_RAW_FIELD:
                    # Virtual field: raw tag_list from HF dataset.
                    val = row.get("tag_list")
                    if val:
                        parts.append(f"{TAG_LIST_RAW_FIELD}: {', '.join(str(x) for x in val)}")
                    continue
                if field == TAG_LIST_CLEAN_FIELD:
                    # Virtual field: cleaned tag_list (lowercase, dedup, noise filter).
                    val = row.get("tag_list")
                    if val:
                        cleaned = _clean_tag_list(val)
                        if cleaned:
                            parts.append(f"{TAG_LIST_CLEAN_FIELD}: {', '.join(cleaned)}")
                    continue
                if field in CRAWL_FIELDS:
                    # Crawl-sourced field: read from JSONL record.
                    jsonl_key = CRAWL_FIELDS[field]
                    val = crawl_rec.get(jsonl_key)
                    if val is None:
                        continue
                    if isinstance(val, list):
                        val = ", ".join(str(x) for x in val)
                    parts.append(f"{field}: {val}")
                    continue
                val = row.get(field)
                if val is None:
                    continue
                if isinstance(val, list):
                    val = ", ".join(str(x) for x in val)
                parts.append(f"{field}: {val}")
            meta_str = "\n".join(parts)
            corpus.append(meta_str)
            metadata_dict[tid] = meta_str

        corpus_tokens = bm25s.tokenize(corpus)
        self.bm25 = bm25s.BM25()
        self.bm25.index(corpus_tokens)

        index_dir.mkdir(parents=True, exist_ok=True)
        self.bm25.save(str(index_dir), corpus=corpus)
        with open(ids_path, "w") as f:
            json.dump(self.track_ids, f)
        with open(meta_path, "w") as f:
            json.dump(metadata_dict, f, ensure_ascii=False)
        self._metadata_dict = metadata_dict

    def _track_id_to_metadata(self, track_id: str) -> str:
        """Map a track_id to its indexed metadata string — used to inject
        music turns from chat_history into the query.
        """
        if self._metadata_dict and track_id in self._metadata_dict:
            return self._metadata_dict[track_id]
        return track_id

    def _maybe_spacy(self, text: str) -> str:
        """Apply spaCy content-token filter when enabled. Used only for
        natural-language fragments (cur_query, user/assistant history
        content). Track metadata blocks bypass this — already keyword-form.
        """
        if not self.spacy_filter or not text:
            return text
        return _spacy_content_tokens(text)

    def _build_query(self, user_query: str, chat_history: list[dict]) -> str:
        """Build the BM25 query string according to query_mode.

        history     — user + assistant + music(meta) + cur_query
        music_only  — music(meta) + cur_query  (default; NL utterances excluded)
        """
        skip_user_assistant = self.query_mode == "music_only"

        parts: list[str] = []
        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "music":
                # metadata block — keyword form, no spaCy filter
                parts.append(self._track_id_to_metadata(content))
            elif role in ("user", "assistant") and not skip_user_assistant:
                parts.append(self._maybe_spacy(content))
        parts.append(self._maybe_spacy(user_query))
        return " ".join(parts)

    def _retrieve_with_scores(self, user_query: str, chat_history: list[dict]) -> list[tuple[str, float]]:
        """Search and return (track_id, bm25_score) pairs with the seen filter applied."""
        query_text = self._build_query(user_query, chat_history)
        query_tokens = bm25s.tokenize([query_text.lower()])
        # Search with headroom so the post-filter result still yields
        # n_candidates unseen track_ids (the session-seen filter is lossless:
        # GT never repeats within a session).
        seen = extract_session_seen_tracks(chat_history)
        search_k = min(self.n_candidates + len(seen), len(self.track_ids))
        results, scores = self.bm25.retrieve(query_tokens, k=search_k)
        # bm25s results/scores are (n_query=1, k) numpy arrays; use row 0.
        pairs = [(self.track_ids[int(idx)], float(score)) for idx, score in zip(results[0], scores[0])]
        return take_unseen_pairs(pairs, seen, self.n_candidates)

    def generate(
        self,
        user_query: str,
        chat_history: list[dict],
        user_id: str,
        conversation_goal: dict | None = None,
        goal_progress: list[dict] | None = None,
        user_profile: dict | None = None,
    ) -> list[str]:
        return [tid for tid, _ in self._retrieve_with_scores(user_query, chat_history)]

    def generate_with_scores(
        self,
        user_query: str,
        chat_history: list[dict],
        user_id: str,
        conversation_goal: dict | None = None,
        goal_progress: list[dict] | None = None,
        user_profile: dict | None = None,
    ) -> list[tuple[str, float]]:
        """Expose native BM25 scores (bypasses the rank-derived fallback)."""
        return self._retrieve_with_scores(user_query, chat_history)
