import asyncio
import subprocess
import time

import psutil

from src.pipeline.analyzer import evaluate_fact
from src.pipeline.extractor import extract_claims_from_chunks
from src.pipeline.retriever import get_relevant_evidence
from src.pipeline.transcriber import get_overlapping_chunks

_BENCHMARK_URL = "https://www.youtube.com/watch?v=sNhhvQGsMEc"
_BOX_WIDTH = 96
_HEAVY = "=" * _BOX_WIDTH
_LIGHT = "-" * _BOX_WIDTH
_LABEL_WIDTH = 34
_VRAM_AVAILABLE = True


def _ram_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def _vram_mb() -> float:
    global _VRAM_AVAILABLE
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return float(result.stdout.strip().split("\n")[0])
    except Exception:
        _VRAM_AVAILABLE = False
        return 0.0


def _row(label: str, value: str) -> str:
    return f"  {label:<{_LABEL_WIDTH}}{value}"


async def run_benchmark():
    print(_HEAVY)
    print("  PRAMAAN — SYSTEM BENCHMARK RUNNER")
    print(f"  Target : {_BENCHMARK_URL}")
    print(_HEAVY)

    print("\n[STAGE 1]  Ingesting transcript and building sliding windows...")
    t_start = time.perf_counter()
    chunks = await get_overlapping_chunks(_BENCHMARK_URL)
    transcript_time = time.perf_counter() - t_start
    print(f"[STAGE 1]  Complete — {len(chunks)} windows  |  {transcript_time:.3f} s")

    print("\n[STAGE 2]  Extracting and deduplicating claims via Qwen 2.5 3B...")
    t_start = time.perf_counter()
    claims = await extract_claims_from_chunks(chunks, _BENCHMARK_URL)
    extraction_time = time.perf_counter() - t_start
    print(f"[STAGE 2]  Complete — {len(claims)} claims  |  {extraction_time:.3f} s")

    if not claims:
        print("\n  No verifiable claims extracted — pipeline halted at Stage 3.")
        print(_HEAVY)
        return

    evidence_times: list[float] = []
    evidence_counts: list[int] = []
    nli_times: list[float] = []
    verdicts: list[str] = []
    confidence_scores: list[float] = []
    ram_before_readings: list[float] = []
    ram_after_readings: list[float] = []
    vram_before_readings: list[float] = []
    vram_after_readings: list[float] = []

    for idx, claim in enumerate(claims, start=1):
        print(f"\n[STAGE 3]  Claim [{idx}/{len(claims)}] — querying DuckDuckGo + FAISS gate...")
        t_start = time.perf_counter()
        evidence_items = await get_relevant_evidence(claim.claim_text)
        ev_time = time.perf_counter() - t_start
        evidence_times.append(ev_time)
        evidence_counts.append(len(evidence_items))
        print(
            f"[STAGE 3]  Claim [{idx}/{len(claims)}] — "
            f"{len(evidence_items)} sentences cleared threshold  |  {ev_time:.3f} s"
        )

        print(f"[STAGE 4]  Claim [{idx}/{len(claims)}] — running DeBERTa NLI cross-encoder...")
        ram_pre = _ram_mb()
        vram_pre = _vram_mb()
        t_start = time.perf_counter()
        result = await evaluate_fact(claim, evidence_items)
        nli_time = time.perf_counter() - t_start
        ram_post = _ram_mb()
        vram_post = _vram_mb()

        nli_times.append(nli_time)
        verdicts.append(result["verdict_label"])
        confidence_scores.append(result["composite_confidence_score"])
        ram_before_readings.append(ram_pre)
        ram_after_readings.append(ram_post)
        vram_before_readings.append(vram_pre)
        vram_after_readings.append(vram_post)

        ram_delta = ram_post - ram_pre
        vram_delta = vram_post - vram_pre
        vram_display = f"{vram_delta:+.1f} MB" if _VRAM_AVAILABLE else "N/A"
        print(
            f"[STAGE 4]  Claim [{idx}/{len(claims)}] — "
            f"{result['verdict_label']:<15}  conf {result['composite_confidence_score']:.2f}  |  "
            f"{nli_time:.3f} s  |  RAM {ram_delta:+.1f} MB  |  VRAM {vram_display}"
        )

    total_evidence_time = sum(evidence_times)
    total_nli_time = sum(nli_times)
    pipeline_total = transcript_time + extraction_time + total_evidence_time + total_nli_time
    peak_ram_delta = max(r_post - r_pre for r_pre, r_post in zip(ram_before_readings, ram_after_readings))
    total_vram_delta = sum(v_post - v_pre for v_pre, v_post in zip(vram_before_readings, vram_after_readings))

    if _VRAM_AVAILABLE:
        gpu_status = (
            f"CLEAN — DeBERTa held on CPU, RTX 2050 unaffected  (cumulative delta {total_vram_delta:+.1f} MB)"
            if total_vram_delta == 0.0
            else f"WARNING — {total_vram_delta:+.1f} MB leaked onto GPU device"
        )
    else:
        gpu_status = "SKIPPED — nvidia-smi not found on PATH"

    print(f"\n\n{_HEAVY}")
    print("  PRAMAAN — BENCHMARK REPORT CARD")
    print(_HEAVY)

    print(_row("STAGE 1  Transcript Ingestion", ""))
    print(_row("  Duration", f"{transcript_time:.3f} s"))
    print(_row("  Sliding Windows Produced", str(len(chunks))))
    print(_LIGHT)

    print(_row("STAGE 2  Claim Extraction  (Qwen 2.5 3B)", ""))
    print(_row("  Duration", f"{extraction_time:.3f} s"))
    print(_row("  Claims Extracted", str(len(claims))))
    print(_LIGHT)

    print(_row("STAGE 3  Evidence Retrieval  (per claim)", ""))
    for i, (ev_t, ev_c) in enumerate(zip(evidence_times, evidence_counts), start=1):
        print(_row(f"  Claim [{i}]", f"{ev_t:.3f} s  |  {ev_c} sentences above cosine threshold"))
    print(_row("  Total Retrieval Time", f"{total_evidence_time:.3f} s"))
    print(_LIGHT)

    print(_row("STAGE 4  NLI Verification  (per claim)", ""))
    for i, (nli_t, verdict, conf, r_pre, r_post, v_pre, v_post) in enumerate(
        zip(
            nli_times,
            verdicts,
            confidence_scores,
            ram_before_readings,
            ram_after_readings,
            vram_before_readings,
            vram_after_readings,
        ),
        start=1,
    ):
        ram_d = r_post - r_pre
        vram_d = v_post - v_pre
        vram_str = f"VRAM {vram_d:+.1f} MB" if _VRAM_AVAILABLE else "VRAM N/A"
        print(
            _row(
                f"  Claim [{i}]",
                f"{nli_t:.3f} s  |  {verdict:<15}  conf {conf:.2f}  |  RAM {ram_d:+.1f} MB  |  {vram_str}",
            )
        )
    print(_row("  Total NLI Time", f"{total_nli_time:.3f} s"))
    print(_row("  Peak Single-Claim RAM Delta", f"{peak_ram_delta:+.1f} MB"))
    print(_row("  Cumulative VRAM Delta", f"{total_vram_delta:+.1f} MB" if _VRAM_AVAILABLE else "N/A"))
    print(_LIGHT)

    print(_row("END-TO-END TOTALS", ""))
    print(_row("  Stage 1  Transcript", f"{transcript_time:.3f} s"))
    print(_row("  Stage 2  Extraction", f"{extraction_time:.3f} s"))
    print(_row("  Stage 3  Evidence", f"{total_evidence_time:.3f} s"))
    print(_row("  Stage 4  NLI", f"{total_nli_time:.3f} s"))
    print(_row("  Pipeline Total", f"{pipeline_total:.3f} s"))
    print(_row("  GPU Status", gpu_status))
    print(_HEAVY)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
