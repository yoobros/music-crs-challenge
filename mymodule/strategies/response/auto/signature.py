"""DSPy signatures for the 'auto' response variant — judge-aligned 3-stage pipeline.

Evaluation framework:
    Composite = 0.50·nDCG@20/20 + 0.10·CatalogDiv + 0.10·Distinct-2 + 0.30·LLM-Judge

Response generation affects the final two terms (Distinct-2 and LLM-Judge).
The Gemini LLM-Judge scores two sub-dimensions on a 1-5 integer scale:

    - Personalization       — how tailored the response feels to THIS user
    - Explanation Quality   — how well each recommendation is justified

These signatures are structured so that the three stages (Personalize →
Explain → Compose) each pre-stage explicit material the Gemini evaluator
rewards, while the final ResponseComposition docstring keeps all three
axes (Personalization / Explanation / Distinct-2) visible to the LLM as
hard constraints.
"""

from __future__ import annotations

try:
    import dspy
except ImportError:  # pragma: no cover
    dspy = None  # type: ignore[assignment]


if dspy is not None:

    class PersonalizationAnalysis(dspy.Signature):
        """Identify how to connect the recommended tracks to THIS specific user.

        The downstream Gemini evaluator scores PERSONALIZATION (1-5): how
        tailored the response feels to the user's query, history, and
        profile. The anchors you produce here become explicit hooks the
        composer threads through the final reply.

        GROUNDING PRIORITY (highest wins)
        ---------------------------------
        1. **Explicit query intent** — the user's current-turn request is
           the strongest signal. Identify its *kind*: mood match / era
           follow / artist extension / lyric-theme / activity context /
           discovery-of-adjacent.
        2. **Conversation history** — prior listens AND prior user turns.
           Name specific prior tracks or topics — not "you listened to
           stuff".
        3. **User profile** — age, country, preferred language / culture.
           Only cite if they add a non-stereotype signal.
        4. **Conversation goal** — its specificity code (HH/HL/LH/LL)
           indicates how tightly to bias toward the stated goal vs free
           exploration.

        RULES
        -----
        - Be concrete. "The user likes rock" is weak; "The user just played
          Pearl Jam's Ten and now asks for other Seattle-scene albums of
          the same era" is anchor-able.
        - If the user query is vague, still commit to a specific axis —
          the composer needs something to hang the response on.
        - NEVER generic ("enjoys music", "has taste"). NEVER stereotypical
          ("Gen-Z so they want TikTok pop", "Korean so K-pop").
        - The opener_hook_seed is a concrete sentence the composer will
          use to begin the reply — make it sound like you're speaking to
          THIS user, not any user.
        """

        user_query: str = dspy.InputField(desc="Current-turn user message.")
        chat_history: str = dspy.InputField(desc="Prior role:content lines; may be empty.")
        user_profile: str = dspy.InputField(desc="Demographics (age, country, language, culture).")
        listener_goal: str = dspy.InputField(desc="Stated listening goal and specificity code.")
        recommended_tracks_overview: str = dspy.InputField(desc="Aggregate summary of the top-20 pool.")

        primary_axis: str = dspy.OutputField(
            desc="One sentence naming the chief personalization angle, e.g. "
            "'era-driven — 1990s Seattle grunge continuation after the user's Pearl Jam listen' OR "
            "'query-specific mood match — melancholic late-night acoustic bed for solo focus'."
        )
        anchors: str = dspy.OutputField(
            desc="3-5 concrete grounding points, one per line, each phrasing a specific connection "
            "(e.g. 'prior listen: Radiohead \"Creep\" → lean into early-90s alt-rock angst' or "
            "'query mentions \"rainy day\" → privilege introspective, acoustic-heavy picks'). "
            "No generic phrasing; every line must cite a concrete handle from query / history / profile."
        )
        opener_hook_seed: str = dspy.OutputField(
            desc="One sentence (≤22 words) the composer can use verbatim or lightly paraphrase as the "
            "opener of the final reply. Must be 2nd-person, reference the user's specific context, and "
            "NOT be a formulaic opener like 'Here are...' or 'I picked...'. Example: "
            "'That late-night acoustic run you started with Iron & Wine keeps unfolding — same hush, "
            "different rooms.'"
        )

    class TrackExplanationPlan(dspy.Signature):
        """For the top-5 tracks, plan the WHY-each-fits reasoning grounded in
        the personalization context.

        The Gemini evaluator also scores EXPLANATION QUALITY (1-5). Strong
        explanations share three traits:

        1. **Concrete** — reference a specific musical attribute (production
           choice, vocal texture, instrumentation, lyrical image, era marker,
           tempo, key, groove).
        2. **Anchored** — tie back to at least one of the personalization
           anchors. The user should feel the pick is for THEM, not generic.
        3. **Varied** — no two tracks share the same opener word or the same
           attribute noun. Distinct-2 depends on this.

        STOCK PHRASE BAN
        ----------------
        Never use: 'perfect for', 'great for', 'you'll love', 'check out',
        'here are', 'this song', 'this track', 'you should'. These tank
        both Explanation Quality AND Distinct-2.

        COVERAGE
        --------
        The pool has 20 tracks; you highlight 5. Produce a single
        `bridging_theme` phrase that names what ties multiple of the 5
        together (and could extend to the tail). Produce a
        `tail_one_liner` that acknowledges the remaining ~15 in one
        evocative phrase without enumerating them.
        """

        primary_axis: str = dspy.InputField(desc="From PersonalizationAnalysis.")
        anchors: str = dspy.InputField(desc="From PersonalizationAnalysis.")
        recommended_tracks_detailed: str = dspy.InputField(
            desc="Top-5 tracks, one per line: 'N. title — artist (album, year) [tags]'."
        )
        recommended_tracks_overview: str = dspy.InputField(
            desc="Aggregate summary of the full 20-track pool (artist counts, year range, tags)."
        )
        query_similarity_hints: str = dspy.InputField(
            desc="Optional. Per-track direct query-semantic relevance annotations. May be empty."
        )
        track_similarity_hints: str = dspy.InputField(
            desc="Optional. Per-track prior-listen similarity annotations. May be empty."
        )

        track_highlights: str = dspy.OutputField(
            desc="3-5 lines, one per selected track. Format per line: "
            "'N. <concrete musical attribute(s)> — <how it echoes one or more anchors>'. "
            "Each line must OPEN with a different word, and NO line may duplicate a content-bearing "
            "noun from another (articles/prepositions excepted). No stock phrase from the ban list above."
        )
        bridging_theme: str = dspy.OutputField(
            desc="One phrase (6-12 words) the composer weaves as connective tissue across the "
            "highlighted tracks AND into the tail. Concrete, not abstract — e.g. "
            "'the same muted guitar and dim-room atmosphere' rather than 'a similar vibe'."
        )
        tail_one_liner: str = dspy.OutputField(
            desc="One phrase (4-10 words) acknowledging the ~15 unhighlighted pool tracks WITHOUT "
            "listing them. Concrete image preferred: 'the rest keeps the same hush', "
            "'everything else bleeds into that same dusk'."
        )

    class ResponseComposition(dspy.Signature):
        """Compose the final assistant reply, optimized for three evaluation axes.

        (1) PERSONALIZATION (LLM-Judge, 1-5)
            Use `opener_hook_seed` as the first sentence (verbatim or lightly
            rephrased). Thread AT LEAST TWO `anchors` into the body by
            paraphrase — not verbatim. The reply must contain nothing that
            could apply to a generic user.

        (2) EXPLANATION QUALITY (LLM-Judge, 1-5)
            For each `track_highlights` line, rewrite it into flowing prose
            while preserving the concrete musical attribute. Weave the
            `bridging_theme` at least twice (once mid-body, once near the
            close). End the body with the `tail_one_liner` to signal
            pool-awareness.

        (3) LEXICAL DIVERSITY / Distinct-2 (evaluator weight 0.10)
            BEFORE writing, commit 5-10 regex patterns to
            `response_excluded_patterns` — the globally-overused stock
            phrases AND 2-3 session-specific vocabulary commitments (verbs
            or adjectives you pledge not to reuse this turn). AFTER writing,
            reread and confirm zero matches.

        TRACK CITATION
        --------------
        You MAY cite tracks by title verbatim, double-quoted ("Title").
        For every title you cite, list it (unquoted, one per line) in
        `cited_titles`. A post-processor fuzzy-matches and either
        canonicalizes to `"<Title>" by <Artist>` or strips as fabrication.
        Prefer grounded descriptive phrases (era / album / artist trait /
        sonic marker) over citations when a natural reference works.

        CRAFT
        -----
        - Vary sentence structure: mix one short declarative, one em-dashed
          aside, one comma-chained list.
        - Specific nouns (hook, bridge, outro, interplay, grain, low-end,
          reverb-tail) over generic (song, track, piece).
        - No two consecutive sentences may start with the same word.
        - End with ONE follow-up question that references a specific pivot
          the user could take next (era shift, artist deep-dive, mood
          counterweight) — NEVER generic ("want more?", "anything else?").

        HARD CONSTRAINTS
        ----------------
        - Plain prose only. No bullets / lists / headings / code fences.
        - Length: 150-220 words.
        - Do NOT reveal pipeline internals (top-k, reranker, PAS,
          intent_group, primary_axis, anchors, themes, etc.).
        - Do NOT open with 'Here are', 'I picked', 'Check out',
          'These tracks' — use opener_hook_seed or a paraphrase.
        """

        user_query: str = dspy.InputField(desc="Current-turn user message.")
        opener_hook_seed: str = dspy.InputField(desc="From PersonalizationAnalysis.")
        anchors: str = dspy.InputField(desc="From PersonalizationAnalysis.")
        track_highlights: str = dspy.InputField(desc="From TrackExplanationPlan.")
        bridging_theme: str = dspy.InputField(desc="From TrackExplanationPlan.")
        tail_one_liner: str = dspy.InputField(desc="From TrackExplanationPlan.")
        recommended_tracks_detailed: str = dspy.InputField(
            desc="Top-5 tracks, `N. title — artist (album, year) [tags]` per line, for grounding "
            "descriptive references."
        )

        response_excluded_patterns: str = dspy.OutputField(
            desc="5-10 python-regex patterns (one per line) that `response` MUST NOT match. "
            "Pre-commit BEFORE writing. Include global stock phrases (\\bthis song\\b, "
            "\\bperfect for\\b, \\byou'?ll love\\b, \\bhere are\\b, \\bcheck out\\b) AND "
            "2-3 session-specific vocabulary commitments."
        )
        cited_titles: str = dspy.OutputField(
            desc="Newline-separated list of titles cited verbatim in `response`. One per line, "
            "UNQUOTED, exact text as cited. Empty string if none."
        )
        response: str = dspy.OutputField(
            desc="Final 150-220 word reply. Opens with opener_hook_seed (or light paraphrase). Weaves "
            "anchors, track_highlights, and bridging_theme (twice). Ends with tail_one_liner then a "
            "specific pivot-referencing follow-up question. Complies with response_excluded_patterns."
        )
