# CoreRAG -- developer tasks.
# Requires: uv (https://docs.astral.sh/uv/). Docker is needed from Checkpoint B onward.
.DEFAULT_GOAL := help

.PHONY: help install install-eval fmt lint type test test-int check up up-obs up-ingest down logs serve ingest eval eval-testset eval-retrieval eval-ragas eval-ablation eval-cache clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Sync the virtualenv from pyproject/uv.lock (lean: no eval-only deps)
	uv sync

install-eval:  ## Sync including the eval group (ragas + friends) [P5]
	uv sync --group eval

fmt:  ## Auto-format and auto-fix lint findings
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Lint (ruff) + verify formatting
	uv run ruff check .
	uv run ruff format --check .

type:  ## Type-check (mypy, strict)
	uv run mypy .

test:  ## Run unit tests (skips integration)
	uv run pytest

test-int:  ## Run integration tests (requires services: make up)
	uv run pytest -m integration

check: lint type test  ## Run all quality gates (lint + type + test)

up:  ## Start core services (qdrant, redis)  [Checkpoint B]
	docker compose up -d qdrant redis

up-obs:  ## Start observability stack (langfuse)  [Checkpoint B]
	docker compose --profile obs up -d

up-ingest:  ## Start ingestion deps (grobid)  [Checkpoint B]
	docker compose --profile ingest up -d

down:  ## Stop all services
	docker compose down

logs:  ## Tail service logs
	docker compose logs -f

serve:  ## Run the API locally with reload  [P0.8]
	uv run uvicorn api.main:app --reload --port 8000

ingest:  ## Run the ingestion pipeline  [P1]
	uv run python -m data.pipeline

eval: eval-retrieval eval-ragas  ## Run the core evaluation suite (retrieval + RAGAS metrics)  [P5]

eval-testset:  ## Generate the golden Q/A set (needs install-eval; costs real $, run once)  [P5.2]
	uv run python -m eval.make_testset

eval-retrieval:  ## Retrieval metrics: hit@k/MRR/nDCG against the golden set  [P5.3]
	uv run python -m eval.retrieval_eval

eval-ragas:  ## RAGAS answer-quality metrics (needs install-eval)  [P5.4]
	uv run python -m eval.ragas_eval

eval-ablation:  ## Contextualization/retrieval ablation table (needs install-eval)  [P5.5]
	uv run python -m eval.ablation

eval-cache:  ## Semantic-cache hit-rate/speedup benchmark  [P5.6]
	uv run python -m eval.latency_bench --cache

clean:  ## Remove tooling caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
