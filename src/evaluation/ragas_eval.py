
import json, time
from loguru import logger
from configs.settings import OUTPUTS

SAMPLE_TEST_SET = [
    {"question": "What are the capital adequacy requirements for UAE banks?", "category": "prudential"},
    {"question": "What AML obligations apply to firms in DIFC?", "category": "aml"},
    {"question": "How does the IMF assess UAE economic diversification?", "category": "macro"},
]

def run_evaluation(rag, test_set=None, use_ragas=False, output_path=None):
    if test_set is None:
        test_set = SAMPLE_TEST_SET
    results = []
    for item in test_set:
        t0 = time.time()
        result = rag.query(item["question"])
        result["latency"] = time.time() - t0
        result["expected_category"] = item.get("category", "")
        results.append(result)
        logger.info(f"Q: {item['question'][:60]}... ({result['latency']:.1f}s)")

    avg_latency = sum(r["latency"] for r in results) / len(results)
    citation_rate = sum(1 for r in results if r["sources"]) / len(results)
    high_conf = sum(1 for r in results if r["confidence"] == "HIGH") / len(results)

    report = {
        "total_questions": len(results),
        "custom_scores": {
            "avg_latency_s": round(avg_latency, 2),
            "citation_rate": round(citation_rate, 3),
            "high_confidence_rate": round(high_conf, 3),
        },
        "results": results,
    }

    out = output_path or OUTPUTS / "evaluation_results.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    logger.success(f"Evaluation saved to {out}")
    return report
