# Evaluating LLM Reliability Under Retrieval Noise in RAG Pipelines

**Yanet Assefa** · Independent Research · 2026

---

## Overview

Real-world RAG systems retrieve imperfect documents. This study evaluates how three types of retrieval noise — irrelevant, contradictory, and misleading — affect LLM reliability across four dimensions: **accuracy**, **hallucination**, **grounding consistency**, and **abstention behavior**.

The goal is to characterize *how* and *when* RAG-based LLMs fail, producing a reproducible benchmark pipeline and documented failure cases.

---

## Key Findings

- Contradictory noise caused the steepest accuracy degradation and highest hallucination rates
- Misleading noise triggered over-abstention — the model refused answerable questions
- Irrelevant noise had minimal effect when the correct document was also present
- **Grounded hallucination** — citing a noisy source to support a wrong answer — was the most dangerous failure pattern identified
- Abstention increased under high-conflict conditions, trading accuracy for coverage

See [`FINDINGS.md`](FINDINGS.md) for the full writeup with methodology, results, and limitations.

---

## Repository Structure

```
rag-reliability-research/
├── src/
│   ├── evaluate.py          # Main evaluation pipeline
│   └── plot_results.py      # Figure generation
├── results/
│   ├── raw_results.csv      # Per-query results (generated on run)
│   └── summary.json         # Aggregated metrics by condition (generated on run)
├── plots/
│   ├── fig1_accuracy_vs_hallucination.png
│   ├── fig2_abstention_rate.png
│   ├── fig3_metrics_heatmap.png
│   └── fig4_accuracy_degradation.png
├── FINDINGS.md              # Full research writeup
├── requirements.txt
└── README.md
```

---

## Experimental Design

### Noise Conditions (7 total)

| Condition | Description |
|---|---|
| Clean (baseline) | Correct document only |
| Irrelevant × 1 | 1 off-topic document added |
| Irrelevant × 3 | 3 off-topic documents added |
| Contradictory × 1 | 1 factually incorrect document contradicting ground truth |
| Contradictory × 2 | 2 conflicting documents |
| Misleading × 1 | 1 document casting doubt on the correct answer |
| Misleading × 2 | 2 misleading documents |

### Evaluation Dimensions

| Metric | Definition |
|---|---|
| **Accuracy** | Response correctly answers the question per ground truth |
| **Hallucination** | Response incorporates false claims from noisy context |
| **Grounding** | Response cites or references retrieved documents |
| **Abstention** | Response expresses uncertainty or declines to answer |

### Model

Claude Haiku (`claude-haiku-4-5`) via Anthropic API. All prompts and configurations are in `src/evaluate.py`.

---

## Quickstart

**Prerequisites:** Python 3.9+, Anthropic API key

```bash
# Clone and install
git clone https://github.com/yanetassefa1/rag-reliability-research
cd rag-reliability-research
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY=your_key_here

# Run evaluation (~5-10 minutes, 56 API calls)
python src/evaluate.py

# Generate figures
python src/plot_results.py
```

Results are written to `results/` and figures to `plots/`.

---

## Results Preview

After running, `results/summary.json` contains aggregated metrics for each noise condition:

```json
{
  "clean_level0": {
    "n": 8,
    "accuracy_rate": 0.875,
    "hallucination_rate": 0.0,
    "grounding_rate": 0.875,
    "abstention_rate": 0.0
  },
  "contradictory_level1": {
    "n": 8,
    "accuracy_rate": 0.5,
    "hallucination_rate": 0.375,
    ...
  }
}
```

---

## Limitations

- Small corpus (8 topics) — findings are directional, not statistically definitive
- Heuristic scoring via keyword overlap — not semantic equivalence
- Single model evaluated
- Synthetic noise, not real retrieval failures
- No mitigation experiments (future work)

---

## Research Context

This work is motivated by safety-relevant questions about LLM behavior under distribution shift. When a model's context conflicts with its training signal, does it maintain reliability or does it follow the retrieved context regardless of accuracy? The grounded hallucination failure mode identified here — where a model cites a noisy source to support a wrong answer — is particularly relevant to deployed RAG systems where users trust citations as indicators of accuracy.

---

## Citation

```
Assefa, Y. (2026). Evaluating LLM Reliability Under Retrieval Noise in RAG Pipelines.
Independent empirical study. https://github.com/yanetassefa1/rag-reliability-research
```

---

*Conducted independently as part of AI research portfolio work. All experiments are reproducible from this repository.*
