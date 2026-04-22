"""
RAG Reliability Evaluation Pipeline
Evaluates LLM behavior under retrieval noise: irrelevant, contradictory, and misleading documents.
"""

import json
import csv
import random
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ─── Data ────────────────────────────────────────────────────────────────────

KNOWLEDGE_BASE = [
    {
        "id": "doc_001",
        "topic": "photosynthesis",
        "text": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of glucose. The reaction occurs in chloroplasts.",
        "answer": "Plants use sunlight, water, and CO2 to produce glucose and oxygen via photosynthesis in chloroplasts.",
    },
    {
        "id": "doc_002",
        "topic": "water_boiling",
        "text": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure (1 atm or 101.325 kPa).",
        "answer": "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    },
    {
        "id": "doc_003",
        "topic": "speed_of_light",
        "text": "The speed of light in a vacuum is approximately 299,792,458 meters per second, commonly approximated as 3 x 10^8 m/s.",
        "answer": "The speed of light is approximately 299,792,458 meters per second.",
    },
    {
        "id": "doc_004",
        "topic": "mitosis",
        "text": "Mitosis is a type of cell division resulting in two daughter cells with the same number of chromosomes as the parent cell. It has four phases: prophase, metaphase, anaphase, and telophase.",
        "answer": "Mitosis produces two daughter cells with identical chromosomes through four phases.",
    },
    {
        "id": "doc_005",
        "topic": "newton_first_law",
        "text": "Newton's first law of motion states that an object at rest stays at rest, and an object in motion stays in motion at constant velocity, unless acted upon by an external force.",
        "answer": "Newton's first law states objects maintain their state of motion unless acted on by an external force.",
    },
    {
        "id": "doc_006",
        "topic": "dna_structure",
        "text": "DNA is a double helix structure composed of nucleotides. Each nucleotide contains a sugar (deoxyribose), a phosphate group, and one of four nitrogenous bases: adenine, thymine, guanine, or cytosine.",
        "answer": "DNA is a double helix made of nucleotides containing deoxyribose sugar, phosphate, and four bases.",
    },
    {
        "id": "doc_007",
        "topic": "gravity",
        "text": "Gravity is a fundamental force that attracts objects with mass toward each other. On Earth, gravity accelerates objects at approximately 9.8 meters per second squared.",
        "answer": "Gravity accelerates objects on Earth at approximately 9.8 m/s squared.",
    },
    {
        "id": "doc_008",
        "topic": "human_heart",
        "text": "The human heart has four chambers: two atria and two ventricles. It pumps blood through the circulatory system, beating approximately 60-100 times per minute in a healthy adult.",
        "answer": "The human heart has four chambers and beats 60-100 times per minute.",
    },
]

NOISE_DOCUMENTS = {
    "irrelevant": [
        "The French Revolution began in 1789 and fundamentally transformed French society.",
        "Shakespeare wrote 37 plays and 154 sonnets during his lifetime.",
        "The Amazon River is the largest river by discharge volume in the world.",
        "Jazz music originated in New Orleans in the early 20th century.",
        "The Great Wall of China stretches over 13,000 miles.",
        "Beethoven composed his Ninth Symphony while completely deaf.",
        "The Eiffel Tower was completed in 1889 and stands 330 meters tall.",
    ],
    "contradictory": {
        "photosynthesis": "Photosynthesis occurs in the mitochondria of plant cells, not in chloroplasts, and produces carbon dioxide as a byproduct.",
        "water_boiling": "Water boils at 50 degrees Celsius at standard atmospheric pressure, which is why it is used in cooking at low temperatures.",
        "speed_of_light": "The speed of light is approximately 150,000 kilometers per second, and it can be slowed significantly in certain media.",
        "mitosis": "Mitosis produces four daughter cells with half the number of chromosomes as the parent cell, which is essential for sexual reproduction.",
        "newton_first_law": "Newton's first law states that all objects naturally decelerate over time due to the inherent resistance of space.",
        "dna_structure": "DNA is a single-stranded molecule composed of ribonucleotides, and it contains the base uracil instead of thymine.",
        "gravity": "On Earth, gravity accelerates objects at approximately 1.6 meters per second squared, the same as on the Moon.",
        "human_heart": "The human heart has three chambers and beats approximately 30-40 times per minute in a healthy adult.",
    },
    "misleading": {
        "photosynthesis": "Recent studies suggest photosynthesis efficiency varies so dramatically across plant species that textbook descriptions may be oversimplified or misleading.",
        "water_boiling": "The boiling point of water is highly variable and context-dependent; some sources report values ranging from 70 to 130 degrees Celsius depending on conditions.",
        "speed_of_light": "The exact speed of light remains a subject of ongoing measurement refinement; some physicists argue current values may need revision.",
        "mitosis": "The classification of mitosis phases is a pedagogical convenience; the actual process is continuous and the phase boundaries are disputed among biologists.",
        "newton_first_law": "Newton's laws are approximations that break down under many real-world conditions; applying them literally can lead to incorrect predictions.",
        "dna_structure": "The Watson-Crick model of DNA structure has been substantially revised; the actual structure is far more complex and variable than the simple double helix.",
        "gravity": "Gravitational acceleration varies significantly across Earth's surface; the commonly cited 9.8 m/s^2 is a rough average that may not apply in many locations.",
        "human_heart": "Heart rate norms have been revised substantially in recent medical literature; the traditional 60-100 bpm range is now considered outdated by many cardiologists.",
    },
}

QUESTIONS = {
    "photosynthesis": "Where does photosynthesis occur in plant cells, and what does it produce?",
    "water_boiling": "At what temperature does water boil at standard atmospheric pressure?",
    "speed_of_light": "What is the speed of light in a vacuum?",
    "mitosis": "What does mitosis produce and how many phases does it have?",
    "newton_first_law": "What does Newton's first law of motion state?",
    "dna_structure": "What is the structure of DNA and what bases does it contain?",
    "gravity": "What is the gravitational acceleration on Earth?",
    "human_heart": "How many chambers does the human heart have and what is the normal heart rate?",
}


# ─── Evaluation ──────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    doc_id: str
    topic: str
    noise_type: str
    noise_level: int
    question: str
    ground_truth: str
    model_response: str
    is_accurate: bool
    is_hallucination: bool
    is_grounded: bool
    is_abstention: bool
    notes: str


def build_context(doc: dict, noise_type: str, noise_level: int) -> str:
    """Build retrieval context with injected noise documents."""
    chunks = [f"Document 1 (Retrieved):\n{doc['text']}"]

    if noise_type == "clean" or noise_level == 0:
        return chunks[0]

    added = 0
    if noise_type == "irrelevant":
        noise_pool = NOISE_DOCUMENTS["irrelevant"]
        random.shuffle(noise_pool)
        for i, noise in enumerate(noise_pool[:noise_level]):
            chunks.append(f"Document {i+2} (Retrieved):\n{noise}")
            added += 1

    elif noise_type == "contradictory":
        contradiction = NOISE_DOCUMENTS["contradictory"].get(doc["topic"])
        if contradiction:
            chunks.insert(0, f"Document 1 (Retrieved):\n{contradiction}")
            chunks[1] = f"Document 2 (Retrieved):\n{doc['text']}"
            added = 1
        for i in range(min(noise_level - 1, len(NOISE_DOCUMENTS["irrelevant"]))):
            chunks.append(f"Document {len(chunks)+1} (Retrieved):\n{NOISE_DOCUMENTS['irrelevant'][i]}")
            added += 1

    elif noise_type == "misleading":
        misleading = NOISE_DOCUMENTS["misleading"].get(doc["topic"])
        if misleading:
            chunks.append(f"Document {len(chunks)+1} (Retrieved):\n{misleading}")
            added = 1
        for i in range(min(noise_level - 1, len(NOISE_DOCUMENTS["irrelevant"]))):
            chunks.append(f"Document {len(chunks)+1} (Retrieved):\n{NOISE_DOCUMENTS['irrelevant'][i]}")
            added += 1

    return "\n\n".join(chunks)


def score_response(response: str, doc: dict, noise_type: str) -> dict:
    """Score a model response on four dimensions."""
    r = response.lower()
    gt = doc["answer"].lower()

    abstention_phrases = [
        "i cannot", "i'm not sure", "i don't know", "unclear",
        "cannot determine", "conflicting", "insufficient", "i am unable",
        "not enough information", "cannot confirm"
    ]
    is_abstention = any(p in r for p in abstention_phrases)

    key_terms = [w for w in gt.split() if len(w) > 4]
    accuracy_score = sum(1 for t in key_terms if t in r) / max(len(key_terms), 1)
    is_accurate = accuracy_score >= 0.4 and not is_abstention

    hallucination_indicators = []
    if noise_type == "contradictory":
        contra = NOISE_DOCUMENTS["contradictory"].get(doc["topic"], "").lower()
        contra_terms = [w for w in contra.split() if len(w) > 5]
        hallucination_indicators = [t for t in contra_terms if t in r and t not in gt]
    is_hallucination = len(hallucination_indicators) >= 2 and not is_abstention

    grounding_phrases = ["according to", "based on", "the document", "retrieved", "states that", "indicates"]
    is_grounded = any(p in r for p in grounding_phrases) or (is_accurate and not is_hallucination)

    return {
        "is_accurate": is_accurate,
        "is_hallucination": is_hallucination,
        "is_grounded": is_grounded,
        "is_abstention": is_abstention,
    }


def query_model(question: str, context: str) -> str:
    """Query Claude with a question and retrieved context."""
    prompt = f"""You are a question-answering assistant. Use the retrieved documents below to answer the question. If the documents contain conflicting information or you are uncertain, say so clearly.

Retrieved Documents:
{context}

Question: {question}

Answer:"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"


def run_evaluation(sample_size: int = None) -> list[EvalResult]:
    """Run the full evaluation across all noise conditions."""
    results = []
    noise_configs = [
        ("clean", 0),
        ("irrelevant", 1),
        ("irrelevant", 3),
        ("contradictory", 1),
        ("contradictory", 2),
        ("misleading", 1),
        ("misleading", 2),
    ]

    docs = KNOWLEDGE_BASE if sample_size is None else random.sample(KNOWLEDGE_BASE, min(sample_size, len(KNOWLEDGE_BASE)))

    total = len(docs) * len(noise_configs)
    count = 0

    for doc in docs:
        question = QUESTIONS[doc["topic"]]
        for noise_type, noise_level in noise_configs:
            count += 1
            print(f"[{count}/{total}] topic={doc['topic']} noise={noise_type} level={noise_level}")

            context = build_context(doc, noise_type, noise_level)
            response = query_model(question, context)
            scores = score_response(response, doc, noise_type)

            result = EvalResult(
                doc_id=doc["id"],
                topic=doc["topic"],
                noise_type=noise_type,
                noise_level=noise_level,
                question=question,
                ground_truth=doc["answer"],
                model_response=response,
                notes=f"context_length={len(context)}",
                **scores,
            )
            results.append(result)
            time.sleep(0.5)

    return results


def save_results(results: list[EvalResult], path: str = "results/raw_results.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        writer.writerows([asdict(r) for r in results])
    print(f"Saved {len(results)} results to {path}")


def compute_summary(results: list[EvalResult]) -> dict:
    """Compute aggregate metrics by noise condition."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        key = f"{r.noise_type}_level{r.noise_level}"
        groups[key].append(r)

    summary = {}
    for condition, group in groups.items():
        n = len(group)
        summary[condition] = {
            "n": n,
            "accuracy_rate": round(sum(r.is_accurate for r in group) / n, 3),
            "hallucination_rate": round(sum(r.is_hallucination for r in group) / n, 3),
            "grounding_rate": round(sum(r.is_grounded for r in group) / n, 3),
            "abstention_rate": round(sum(r.is_abstention for r in group) / n, 3),
        }
    return summary


if __name__ == "__main__":
    print("Starting RAG Reliability Evaluation...")
    print("=" * 60)
    results = run_evaluation()
    save_results(results)
    summary = compute_summary(results)
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary by noise condition:")
    for condition, metrics in summary.items():
        print(f"\n{condition}:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    print("\nDone. Run `python src/plot_results.py` to generate figures.")
