"""Evaluación automática del agente F1-GPT.

Reproduce las métricas reportadas en la sección 6.4 de la memoria:
  - Task completion rate (proporción de respuestas válidas)
  - Precisión por categoría (descriptive / predictive / comparative)
  - Latencia extremo a extremo (media, p50, p90)
  - Coste estimado por consulta (USD) con las tarifas públicas de Gemini

Uso:
    python evaluate.py [--limit 5] [--output evaluation/results.json]

El banco de consultas está en evaluation/benchmark.json y se actualiza
manualmente con la respuesta esperada para nuevas pruebas.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime
from typing import Any

# Tarifas Gemini 2.0 Flash (USD por millón de tokens, ver memoria sección 5.3)
GEMINI_PRICE_INPUT = 0.10
GEMINI_PRICE_OUTPUT = 0.40

# Estimación gruesa de tokens cuando no podemos contarlos exactamente.
# Regla habitual: ~4 caracteres por token en inglés / español.
CHARS_PER_TOKEN = 4

DEFAULT_BENCHMARK = os.path.join("evaluation", "benchmark.json")
DEFAULT_OUTPUT = os.path.join("evaluation", "results.json")


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _check_keywords(response: str, keywords: list[str]) -> bool:
    """Devuelve True si al menos una de las palabras esperadas está presente."""
    if not keywords:
        return True
    response_lower = response.lower()
    return any(k.lower() in response_lower for k in keywords)


def _classify_outcome(case: dict[str, Any], response: str, error: str | None) -> str:
    """Etiqueta el resultado: ok / wrong_keywords / refused / error."""
    if error:
        return "error"
    if case["category"] == "out_of_domain":
        # En out_of_domain la respuesta correcta es un rechazo informado.
        return "ok" if _check_keywords(response, case.get("expected_keywords", [])) else "wrong_keywords"
    if not response or not response.strip():
        return "error"
    if _check_keywords(response, case.get("expected_keywords", [])):
        return "ok"
    return "wrong_keywords"


def _load_agent():
    """Importa el agente sólo si se va a ejecutar de verdad."""
    from src.agent.f1_agent import F1Agent

    return F1Agent()


def run_benchmark(
    benchmark_path: str = DEFAULT_BENCHMARK,
    output_path: str = DEFAULT_OUTPUT,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    with open(benchmark_path, "r", encoding="utf-8") as fh:
        bench = json.load(fh)

    cases = bench["queries"]
    if limit:
        cases = cases[:limit]

    agent = None if dry_run else _load_agent()

    results: list[dict[str, Any]] = []
    for case in cases:
        prompt = case["query"]
        t0 = time.perf_counter()
        response = ""
        error: str | None = None
        try:
            if dry_run:
                response = "[dry-run] respuesta simulada"
            else:
                response = agent.handle_query(prompt)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        latency = time.perf_counter() - t0

        tokens_in = _estimate_tokens(prompt)
        tokens_out = _estimate_tokens(response)
        cost_usd = (
            tokens_in / 1_000_000 * GEMINI_PRICE_INPUT
            + tokens_out / 1_000_000 * GEMINI_PRICE_OUTPUT
        )

        outcome = _classify_outcome(case, response, error)
        results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "query": prompt,
                "response": response,
                "error": error,
                "outcome": outcome,
                "latency_s": round(latency, 3),
                "tokens_in_est": tokens_in,
                "tokens_out_est": tokens_out,
                "cost_usd_est": round(cost_usd, 6),
            }
        )

    summary = _summarise(results)
    payload = {
        "executed_at": datetime.utcnow().isoformat() + "Z",
        "benchmark": benchmark_path,
        "n_queries": len(results),
        "summary": summary,
        "results": results,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    _print_summary(summary, len(results))
    print(f"\nResultados guardados en {output_path}")
    return payload


def _summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {}

    by_category: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)

    def rate(rows: list[dict[str, Any]], outcome: str) -> float:
        return round(100 * sum(r["outcome"] == outcome for r in rows) / max(len(rows), 1), 1)

    latencies = sorted(r["latency_s"] for r in results)
    p50 = latencies[len(latencies) // 2]
    p90_idx = int(0.9 * (len(latencies) - 1))
    p90 = latencies[p90_idx]

    return {
        "task_completion_rate_%": rate(results, "ok"),
        "by_category_ok_%": {cat: rate(rows, "ok") for cat, rows in by_category.items()},
        "latency_mean_s": round(statistics.mean(r["latency_s"] for r in results), 3),
        "latency_p50_s": p50,
        "latency_p90_s": p90,
        "cost_total_usd": round(sum(r["cost_usd_est"] for r in results), 5),
        "cost_per_query_usd_mean": round(
            statistics.mean(r["cost_usd_est"] for r in results), 6
        ),
    }


def _print_summary(summary: dict[str, Any], n: int) -> None:
    print("\n=== Resumen de evaluación ===")
    print(f"Consultas ejecutadas: {n}")
    print(f"Task completion rate: {summary.get('task_completion_rate_%')}%")
    by_cat = summary.get("by_category_ok_%", {})
    if by_cat:
        print("Aciertos por categoría:")
        for cat, v in by_cat.items():
            print(f"  - {cat}: {v}%")
    print(f"Latencia media: {summary.get('latency_mean_s')} s "
          f"(p50 {summary.get('latency_p50_s')} s · p90 {summary.get('latency_p90_s')} s)")
    print(f"Coste total estimado: ${summary.get('cost_total_usd')} "
          f"(media {summary.get('cost_per_query_usd_mean')} $/consulta)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None,
                        help="Ejecuta sólo las primeras N consultas (útil para humo).")
    parser.add_argument("--dry-run", action="store_true",
                        help="No invoca el LLM; verifica que el banco y el evaluador funcionan.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_benchmark(
        benchmark_path=args.benchmark,
        output_path=args.output,
        limit=args.limit,
        dry_run=args.dry_run,
    )
