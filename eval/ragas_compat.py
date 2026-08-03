"""RAGAS import/wiring compatibility shim (P5.1).

Must be imported (directly or transitively) before any other ``ragas`` import in
this codebase.

Confirmed live 2026-08-03: bare `import ragas` crashes on both the latest release
(0.4.3) and the previously-"safe" 0.3.9 -- `ragas/llms/base.py` unconditionally
imports `langchain_community.chat_models.vertexai.ChatVertexAI` at module load, but
our `langchain-core>=1.4` pin (needed for P3's agent graph) forces `langchain-community`
to resolve to a version that already dropped that submodule (it's being sunset;
VertexAI support moved to a standalone package). This is a confirmed open upstream
bug (github.com/vibrantlabsai/ragas/issues/2745, unmerged fix in #2793) -- pinning
ragas lower doesn't help, since the *langchain-community* version is what's broken,
and nothing in ragas constrains it. VertexAI support itself is irrelevant to us (we
use OpenAI); the fix below just satisfies the import, nothing more.

Second confirmed live finding: every ragas LLM call (``LangchainLLMWrapper.agenerate_text``)
unconditionally calls ``ragas._analytics.track()`` at the end, which does a *synchronous*
``requests.post`` to ragas's own telemetry endpoint (t.explodinggradients.com) from inside
an ``async def`` -- blocking the whole event loop. That host is slow/unreachable from this
network: measured live, every single call (even a trivial "Say OK.") took a consistent
~31s, vs. ~0.7-1.2s for the exact same model called directly via our own
``core.llm.get_agent_model`` (bypassing ragas). Setting the library's own documented
opt-out (``RAGAS_DO_NOT_TRACK=True``) drops it back to ~1s -- confirmed live, a ~30x
fix. Set here, once, before ragas reads it.
"""

from __future__ import annotations

import os
import sys
import types
from typing import cast

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "True")

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class _StubChatVertexAI:
        """Placeholder satisfying ragas's unconditional import; never instantiated."""

    _vertexai_stub.ChatVertexAI = _StubChatVertexAI  # type: ignore[attr-defined]
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

from langchain_core.embeddings import Embeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.llms.base import BaseRagasLLM
from ragas.run_config import RunConfig

from core.config import Settings
from core.llm import get_agent_model
from data.embeddings import embed_dense

# Third confirmed live finding: LangchainLLMWrapper.set_run_config narrows
# RunConfig's default broad exception_types=(Exception,) down to just
# openai.RateLimitError whenever the wrapped model is OpenAI -- so a transient
# openai.APIConnectionError (confirmed live: recurs frequently in this network's
# concurrent-connection bursts, not a one-off) isn't retried internally at all.
# Lowering max_workers reduces how many connections fire at once, which reduces
# how often the burst triggers a drop in the first place; callers (make_testset.py)
# add their own outer retry around the whole multi-call pipeline as a backstop,
# since ragas exposes no per-call retry hook we can reach from outside.
_RUN_CONFIG = RunConfig(max_workers=4)


class LocalEmbeddingsAdapter(Embeddings):
    """Wraps our local FastEmbed dense embedder as a LangChain ``Embeddings``.

    Same "wrap our own local model in the library's expected interface" pattern as
    P4's RedisVL ``CustomVectorizer`` -- no new paid embedding provider for eval.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_dense(texts, self._settings)

    def embed_query(self, text: str) -> list[float]:
        return embed_dense([text], self._settings)[0]


def ragas_llm(settings: Settings) -> BaseRagasLLM:
    """RAGAS judge: our existing agent model (gpt-4o-mini) -- no new provider."""
    # ragas ships no type stubs (ignore_missing_imports); the wrapper genuinely
    # returns a BaseRagasLLM at runtime, cast rather than widen the return type.
    return cast(
        BaseRagasLLM, LangchainLLMWrapper(get_agent_model(settings), run_config=_RUN_CONFIG)
    )


def ragas_embeddings(settings: Settings) -> BaseRagasEmbeddings:
    """RAGAS embedder: our existing local dense embedder -- no new provider."""
    return cast(BaseRagasEmbeddings, LangchainEmbeddingsWrapper(LocalEmbeddingsAdapter(settings)))
