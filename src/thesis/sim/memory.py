"""The memory stream: what a persona "remembers" when it writes a reply.

Follows Park et al.'s generative-agent memory architecture, which retrieves
from a store of past observations by combining three signals:

    score = w_recency * recency + w_importance * importance + w_relevance * relevance

with ``recency = decay ** hours_elapsed``. Each component is min-max normalized
to [0, 1] before weighting, because they are otherwise on incomparable scales
and whichever happened to have the widest raw range would dominate the sum.

**Named scope cuts**, stated here rather than left for a reader to notice:

1. *No planning module, no day-loop, no environment.* Park et al. simulate
   agents living through days in a world. This study asks a single-turn
   question -- given this message, what do you reply? -- so the planning and
   environment machinery would be elaborate scaffolding around a stimulus that
   does not need it.
2. *One reflection level, not recursive.* Reflections synthesize observations,
   but reflections never reflect on reflections. Deeper recursion mainly
   matters over long simulated lifetimes, which cut 1 has already removed.
3. *Lexical relevance, not embedding relevance.* Park et al. score relevance by
   embedding cosine similarity. This uses TF-IDF cosine instead. The honest
   trade: TF-IDF matches words rather than meaning, so "settlement date" and
   "payment deadline" do not match, and retrieval is correspondingly cruder.
   It is used because the candidate set per persona is small (tens of items,
   not thousands), which is the regime where lexical overlap degrades least,
   and because an embedding model is a heavy dependency for a component that
   conditions a prompt rather than producing a measured result.
   :func:`retrieve` takes the scorer as an argument, so substituting embeddings
   later changes one call site rather than this module.

**Memory is generated per (persona x direction), not per scenario.** The same
remembered history is reused across every scenario, stakes level, and replicate
sharing that pair. That is both cheaper and a cleaner control: when two cells
differ only in stakes, the memory is held constant by construction, so a
difference between them cannot be an artifact of different remembered history.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Park et al.'s exponential recency decay, applied per hour of simulated
# elapsed time.
RECENCY_DECAY = 0.995

# Equal weighting across the three signals, matching the reference
# implementation. Frozen here rather than tuned: there is no held-out criterion
# in this study that tuning them could honestly optimize against.
WEIGHT_RECENCY = 1.0
WEIGHT_IMPORTANCE = 1.0
WEIGHT_RELEVANCE = 1.0

TOP_K = 8

# Reflection triggers every N observations and produces M synthesized
# statements, per the plan's "one reflection level" cut.
REFLECTION_EVERY = 5
REFLECTIONS_PER_TRIGGER = 3

RelevanceScorer = Callable[[str, Sequence[str]], list[float]]


@dataclass(frozen=True, slots=True)
class MemoryItem:
    """One remembered thing: an observation, or a reflection over observations.

    ``hours_ago`` is simulated elapsed time, not wall-clock. The study has no
    real timeline -- these are archetypes, not agents living through dates --
    so recency is expressed directly as "how long ago did this happen for this
    persona", which is the only sense in which it is meaningful here.
    """

    text: str
    importance: float
    hours_ago: float
    kind: str = "observation"

    @property
    def is_reflection(self) -> bool:
        return self.kind == "reflection"


def _normalize(values: Sequence[float]) -> list[float]:
    """Min-max to [0, 1]; a constant vector maps to all-ones.

    All-ones rather than all-zeros for the constant case: if every candidate
    scores identically on a signal, that signal carries no information and
    should not veto items, which is what zeroing it would do.
    """
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [1.0] * len(values)
    span = high - low
    return [(v - low) / span for v in values]


def tfidf_relevance(query: str, documents: Sequence[str]) -> list[float]:
    """Cosine similarity between the query and each document, over TF-IDF.

    Returns zeros when the vocabulary does not overlap at all, which is the
    correct answer rather than an error: it simply means no remembered item
    shares wording with the situation, and the other two signals decide.
    """
    if not documents:
        return []
    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform([query, *documents])
    except ValueError:
        # Raised when every input is a stop word or empty, leaving an empty
        # vocabulary. No overlap is expressible, so nothing is relevant.
        return [0.0] * len(documents)
    similarities = cosine_similarity(matrix[0:1], matrix[1:])[0]
    return [float(value) for value in similarities]


def recency_scores(items: Sequence[MemoryItem], decay: float = RECENCY_DECAY) -> list[float]:
    """Exponential decay in elapsed hours."""
    return [decay**item.hours_ago for item in items]


def retrieve(
    items: Sequence[MemoryItem],
    query: str,
    *,
    top_k: int = TOP_K,
    relevance: RelevanceScorer = tfidf_relevance,
    decay: float = RECENCY_DECAY,
) -> list[MemoryItem]:
    """The ``top_k`` most retrievable memories for this situation.

    Ties break toward the earlier item in the input, so retrieval is fully
    deterministic: the same store and query always yield the same memories, in
    the same order, which is required for a cached prompt to stay byte-stable.
    """
    if not items:
        return []

    combined = [
        WEIGHT_RECENCY * rec + WEIGHT_IMPORTANCE * imp + WEIGHT_RELEVANCE * rel
        for rec, imp, rel in zip(
            _normalize(recency_scores(items, decay)),
            _normalize([item.importance for item in items]),
            _normalize(relevance(query, [item.text for item in items])),
            strict=True,
        )
    ]
    ranked = sorted(enumerate(items), key=lambda pair: (-combined[pair[0]], pair[0]))
    return [item for _, item in ranked[:top_k]]


def reflection_batches(
    observations: Sequence[MemoryItem], every: int = REFLECTION_EVERY
) -> list[list[MemoryItem]]:
    """Group observations into the batches each reflection is drawn from.

    A trailing partial batch is dropped rather than reflected over: reflecting
    across two observations produces a statement with far weaker support than
    one drawn from five, and mixing the two would put memories of unequal
    evidential weight into the same store with no way to tell them apart.
    """
    if every <= 0:
        msg = "reflection batch size must be positive"
        raise ValueError(msg)
    batches = [list(observations[i : i + every]) for i in range(0, len(observations), every)]
    return [batch for batch in batches if len(batch) == every]


def build_reflection_prompt(batch: Sequence[MemoryItem], n_statements: int) -> str:
    """Ask the model to synthesize higher-level statements from observations."""
    lines = "\n".join(f"- {item.text}" for item in batch)
    return (
        f"Here are things that happened in your working life recently:\n\n{lines}\n\n"
        f"Write {n_statements} short, general statements about how you operate "
        f"at work that these observations support. Each should be one sentence, "
        f"stated as a habit or tendency rather than as a single event. Do not "
        f"restate the observations."
    )


def as_reflection(item: MemoryItem) -> MemoryItem:
    """Mark an item as a reflection.

    Reflections inherit the recency of the batch they came from, so a
    reflection over old observations does not present itself as a fresh
    memory -- which would let stale material outrank recent observations purely
    by virtue of having been synthesized.
    """
    return replace(item, kind="reflection")


def render_memory_block(items: Sequence[MemoryItem]) -> str:
    """The memory section of the prompt.

    Reflections are labelled separately from observations because they are
    different kinds of claim: one is something that happened, the other is a
    generalization the persona holds about itself.
    """
    if not items:
        return ""

    observations = [i for i in items if not i.is_reflection]
    reflections = [i for i in items if i.is_reflection]

    parts = ["## What you have in mind", ""]
    if reflections:
        parts.append("How you tend to operate:")
        parts.extend(f"- {item.text}" for item in reflections)
        parts.append("")
    if observations:
        parts.append("Recent context:")
        parts.extend(f"- {item.text}" for item in observations)
    return "\n".join(parts).rstrip()
