import json
import sys
from collections import defaultdict
from search_engine import HybridLegalAssistant

# Suppress print statements during evaluation
import io
from contextlib import redirect_stdout


def normalize_article_id(doc_id):
    """
    Normalize article ID to article level (remove fıkra/paragraph suffix).
    E.g., "LAW:6098-12-1" -> "LAW:6098-12"
    """
    if doc_id.startswith("LAW:"):
        parts = doc_id.split("-")
        # Keep LAW: and first two parts (law_code and article number)
        if len(parts) >= 3:
            return "-".join(parts[:2])
    return doc_id


def calculate_hit_at_k(retrieved_ids, ground_truth_ids, k=5):
    """
    Calculate Hit@K: Is at least one correct article found in top-k?
    Returns 1 if yes, 0 if no.
    Handles both full IDs (LAW:6098-12-1) and article-level IDs (LAW:6098-12).
    """
    top_k_ids = retrieved_ids[:k]
    ground_truth_set = set(ground_truth_ids)
    
    # Check if any retrieved ID matches any ground truth ID
    # Match can be exact or at article level
    for retrieved_id in top_k_ids:
        # Normalize to article level for comparison
        normalized_id = normalize_article_id(retrieved_id)
        if normalized_id in ground_truth_set or retrieved_id in ground_truth_set:
            return 1
        # Also check if ground truth is a prefix of retrieved (e.g., LAW:6098-12 matches LAW:6098-12-1)
        for gt_id in ground_truth_set:
            if retrieved_id.startswith(gt_id + "-") or gt_id.startswith(normalized_id + "-"):
                return 1
    
    return 0


def calculate_precision_at_k(retrieved_ids, ground_truth_ids, k=5):
    """
    Calculate Precision@K: How many of the top k are relevant?
    Returns precision value between 0 and 1.
    Handles both full IDs (LAW:6098-12-1) and article-level IDs (LAW:6098-12).
    """
    if k == 0:
        return 0.0
    
    top_k_ids = retrieved_ids[:k]
    ground_truth_set = set(ground_truth_ids)
    
    relevant_count = 0
    for retrieved_id in top_k_ids:
        normalized_id = normalize_article_id(retrieved_id)
        # Check if retrieved ID matches any ground truth ID
        if normalized_id in ground_truth_set or retrieved_id in ground_truth_set:
            relevant_count += 1
        else:
            # Check prefix matching
            for gt_id in ground_truth_set:
                if retrieved_id.startswith(gt_id + "-") or gt_id.startswith(normalized_id + "-"):
                    relevant_count += 1
                    break
    
    return relevant_count / k


def evaluate_benchmark(benchmark_file="data/multi_law_benchmark.json", top_k=5):
    """
    Evaluate the retrieval system on the multi-domain benchmark.
    """
    print("=" * 60)
    print("MULTI-LAW SYSTEM EVALUATION")
    print("=" * 60)
    
    # Load benchmark
    print(f"\n[1/3] Loading benchmark from {benchmark_file}...")
    try:
        with open(benchmark_file, "r", encoding="utf-8") as f:
            benchmark = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Benchmark file not found: {benchmark_file}")
        print("Please run generate_benchmark.py first to create the benchmark.")
        return
    
    test_cases = benchmark["test_cases"]
    print(f"  -> Loaded {len(test_cases)} test cases")
    
    # Initialize system
    print(f"\n[2/3] Initializing HybridLegalAssistant...")
    # Suppress initialization prints
    with redirect_stdout(io.StringIO()):
        assistant = HybridLegalAssistant()
    print("  -> System initialized")
    
    # Evaluate each test case
    print(f"\n[3/3] Evaluating {len(test_cases)} test cases...")
    
    # Store results per topic
    topic_results = defaultdict(lambda: {
        "hits": [],
        "precisions": [],
        "total": 0
    })
    
    # Process each test case
    for idx, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        topic = test_case["topic"]
        ground_truth = test_case["ground_truth"]
        
        # Suppress search prints during evaluation
        with redirect_stdout(io.StringIO()):
            retrieved_ids = assistant.search_statutes(query, top_k=top_k)
        
        # Calculate metrics
        hit = calculate_hit_at_k(retrieved_ids, ground_truth, k=top_k)
        precision = calculate_precision_at_k(retrieved_ids, ground_truth, k=top_k)
        
        # Store results
        topic_results[topic]["hits"].append(hit)
        topic_results[topic]["precisions"].append(precision)
        topic_results[topic]["total"] += 1
        
        # Progress indicator
        if idx % 10 == 0:
            print(f"  -> Processed {idx}/{len(test_cases)} test cases...")
    
    # Calculate aggregate metrics per topic
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # Print results table
    print("\nPerformance by Law Category:")
    print("-" * 60)
    print(f"{'Category':<20} {'Cases':<8} {'Hit@5':<12} {'Precision@5':<15}")
    print("-" * 60)
    
    overall_hits = []
    overall_precisions = []
    
    # Sort topics for consistent output
    sorted_topics = sorted(topic_results.keys())
    
    for topic in sorted_topics:
        results = topic_results[topic]
        hits = results["hits"]
        precisions = results["precisions"]
        total = results["total"]
        
        hit_rate = sum(hits) / len(hits) if hits else 0.0
        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
        
        overall_hits.extend(hits)
        overall_precisions.extend(precisions)
        
        print(f"{topic:<20} {total:<8} {hit_rate*100:>6.2f}%    {avg_precision*100:>6.2f}%")
    
    # Overall metrics
    print("-" * 60)
    overall_hit_rate = sum(overall_hits) / len(overall_hits) if overall_hits else 0.0
    overall_avg_precision = sum(overall_precisions) / len(overall_precisions) if overall_precisions else 0.0
    print(f"{'OVERALL':<20} {len(overall_hits):<8} {overall_hit_rate*100:>6.2f}%    {overall_avg_precision*100:>6.2f}%")
    print("=" * 60)
    
    # Detailed breakdown (optional, can be commented out for cleaner output)
    print("\nDetailed Metrics:")
    print("-" * 60)
    for topic in sorted_topics:
        results = topic_results[topic]
        hits = results["hits"]
        precisions = results["precisions"]
        total = results["total"]
        
        hit_count = sum(hits)
        hit_rate = hit_count / total if total > 0 else 0.0
        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
        
        print(f"\n{topic}:")
        print(f"  Total Cases: {total}")
        print(f"  Hits: {hit_count}/{total} ({hit_rate*100:.2f}%)")
        print(f"  Average Precision@5: {avg_precision:.4f}")
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    
    return {
        "overall_hit_rate": overall_hit_rate,
        "overall_avg_precision": overall_avg_precision,
        "topic_results": dict(topic_results)
    }


def main():
    benchmark_file = "data/multi_law_benchmark.json"
    top_k = 5
    
    if len(sys.argv) > 1:
        benchmark_file = sys.argv[1]
    if len(sys.argv) > 2:
        top_k = int(sys.argv[2])
    
    evaluate_benchmark(benchmark_file, top_k=top_k)


if __name__ == "__main__":
    main()

