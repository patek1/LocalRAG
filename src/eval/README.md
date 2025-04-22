# Evaluation

This sub-package contains the Python modules responsible for evaluating the performance of the RAG pipeline.

## Modules:

-   **`metrics_retrieval.py`**: Calculates retrieval quality metrics (NDCG@10, Precision@10, Recall@10) using the `ranx` library based on the provided ground truth (qrels).
-   **`metrics_generation.py`**: Calculates generation quality metrics.
    -   Computes ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) for answerable questions using the `evaluate` library.
    -   Calculates the accuracy of identifying unanswerable questions based on the model's output.
-   **`metrics_ragas.py`**: Implements evaluation using the `RAGAs` framework.
    -   Initializes the evaluator LLM (currently using OpenAI's `gpt-4.1-mini-2025-04-14` via API for performance).
    -   Calculates metrics like Faithfulness and Answer Relevancy for answerable questions.
-   **`metrics_latency.py`**: Calculates and saves performance metrics.
    -   Computes statistical summaries (mean, median, P95, P99) for retrieval and generation latencies.
    -   Handles saving both latency and quality metrics to JSON files in the `results/` directory.