# Findings: LLM Reliability Under Retrieval Noise in RAG Pipelines

*Yanet Assefa — 2026*

---

## Abstract

Retrieval-Augmented Generation (RAG) systems are widely used to ground large language model (LLM) outputs in external knowledge. However, real-world retrieval is imperfect — retrieved documents may be irrelevant, contradictory, or subtly misleading. This study evaluates how retrieval noise affects LLM reliability across four dimensions: accuracy, hallucination, grounding, and abstention. We find that contradictory and misleading noise conditions cause the steepest accuracy degradation and that the model exhibits adaptive abstention behavior under high-noise conditions — a partial mitigation that trades reliability for coverage.

---

## 1. Motivation

Standard RAG evaluations assume retrieved documents are relevant and correct. In practice, retrieval systems return noisy results due to semantic mismatch, outdated content, or adversarial inputs. Understanding how LLMs respond to these failure modes is safety-relevant: a model that confidently generates incorrect answers under noisy context is more dangerous than one that abstains or expresses uncertainty.

This connects to broader alignment concerns. If a model's behavior changes substantially based on what is in its context window — even when that context contradicts its training signal — that is an instance of the reliability and oversight problem that motivated research on scalable oversight and alignment auditing.

---

## 2. Research Questions

1. Does retrieval noise (irrelevant, contradictory, misleading) degrade LLM accuracy, and by how much?
2. Does the model hallucinate under contradictory context, or does it abstain?
3. Is there a grounding consistency tradeoff — does the model cite retrieved documents even when they are wrong?
4. How does noise *level* (number of noisy documents) affect each metric?

---

## 3. Method

### 3.1 Dataset

We constructed a controlled evaluation corpus of 8 factual knowledge items spanning science domains (biology, physics, chemistry). Each item includes: a ground-truth document, a correct answer, and a question. Topics: photosynthesis, water boiling point, speed of light, mitosis, Newton's first law, DNA structure, gravity, human heart anatomy.

### 3.2 Noise Injection

Three noise types were injected into the retrieval context:

| Noise Type | Description |
|---|---|
| **Irrelevant** | Factually correct documents on unrelated topics added to the context |
| **Contradictory** | A document asserting the factually incorrect opposite of the ground truth |
| **Misleading** | A document casting doubt on the ground truth without directly contradicting it |

Noise levels (1–3 additional documents) were tested for each type. The baseline condition (clean, level 0) served as the control.

### 3.3 Evaluation Dimensions

Each model response was scored on four binary dimensions:

- **Accuracy**: Does the response correctly answer the question based on ground truth?
- **Hallucination**: Does the response incorporate false claims sourced from contradictory noise?
- **Grounding**: Does the response cite or reference retrieved documents?
- **Abstention**: Does the response express uncertainty or decline to answer?

### 3.4 Model

Claude Haiku (claude-haiku-4-5) was queried via the Anthropic API with a standard RAG prompt instructing the model to use retrieved documents and to express uncertainty if documents conflicted. Temperature: default. Max tokens: 300.

### 3.5 Sample Size

56 evaluations total: 8 topics × 7 noise conditions (1 clean + 6 noisy).

---

## 4. Results

### 4.1 Accuracy Degradation

Accuracy under the clean baseline was highest across conditions. Contradictory noise caused the steepest single-condition accuracy drop, particularly when the contradictory document appeared first in the retrieved context (simulating rank-1 retrieval failure). Misleading noise — which does not directly contradict but introduces epistemic uncertainty — caused moderate degradation.

Irrelevant noise had a smaller effect on accuracy, suggesting the model is reasonably robust to off-topic retrievals when the correct document is also present.

### 4.2 Hallucination Behavior

Hallucination rates were highest under contradictory noise conditions, as expected. Notably, at higher noise levels, hallucination rates partially decreased as abstention rates increased — the model shifted from confidently wrong to expressing uncertainty. This tradeoff is consistent with the hypothesis that strong epistemic conflict triggers abstention behavior, while mild conflict (one contradictory document) is more likely to produce confident hallucination.

### 4.3 Abstention Patterns

Abstention was near zero under clean and irrelevant conditions. It increased substantially under contradictory and misleading conditions. This is a desirable behavior — the model detecting inconsistency and refusing to commit — but it represents a coverage cost: questions the model could answer correctly under clean retrieval become unanswerable under noisy retrieval.

### 4.4 Grounding Consistency

Grounding rates remained relatively stable across noise conditions. In several failure cases, the model cited retrieved documents *while producing incorrect answers* — grounding its hallucinations in the noisy context. This is the most concerning failure pattern: the model appears well-behaved (cites sources) while being factually wrong.

---

## 5. Failure Case Analysis

**Failure Type 1 — Confident hallucination from contradictory document:**
The model incorporated false claims from a rank-1 contradictory document and answered confidently with no uncertainty expressed. Most common under contradictory-level-1 conditions.

**Failure Type 2 — Grounded hallucination:**
The model cited a retrieved document to support an incorrect answer. This failure mode is particularly problematic because surface indicators of trustworthiness (source citation) were present despite factual error.

**Failure Type 3 — Over-abstention under mild misleading noise:**
Under misleading-level-1 conditions, the model occasionally refused to answer questions it should have been able to answer, citing "conflicting information" when only minor epistemic hedging was present in the retrieved context.

---

## 6. Limitations

- **Small corpus**: 8 topics limits statistical power. Findings are directional, not definitive.
- **Heuristic scoring**: Accuracy and hallucination were scored programmatically using keyword overlap — not semantic equivalence. Some correct paraphrases may be marked inaccurate.
- **Single model**: Results are specific to Claude Haiku and may not generalize to other LLMs or model sizes.
- **Synthetic noise**: Noise documents were hand-crafted, not drawn from real retrieval failures. Real-world noise may have different distributional properties.
- **No mitigation experiments**: This study characterizes failure modes. Mitigation strategies (e.g., confidence-weighted retrieval, chain-of-thought grounding) are left for future work.

---

## 7. Conclusions

Retrieval noise meaningfully degrades LLM reliability, with contradictory and misleading noise being more damaging than irrelevant noise. The model exhibits adaptive abstention under high-conflict conditions, which partially mitigates hallucination but introduces a coverage tradeoff. Grounded hallucination — where the model cites a noisy source to support a wrong answer — represents the most operationally dangerous failure mode identified.

These findings suggest that retrieval quality is a first-order determinant of RAG reliability, and that LLM-level mitigations (e.g., uncertainty prompting) are insufficient to fully compensate for poor retrieval.

---

## 8. Reproducibility

All code, data, and evaluation scripts are available in this repository. To reproduce:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
python src/evaluate.py
python src/plot_results.py
```

Results are saved to `results/` and figures to `plots/`.

---

*This research was conducted as part of an independent empirical study on LLM reliability and RAG system evaluation.*
