# Results

This directory stores the output files generated during the RAG pipeline evaluation runs.

## Contents:

-   **`baseline_latency_metrics.json`**: Contains performance metrics related to the speed of the pipeline components. Includes statistical summaries (mean, median, P95, P99) for:
    -   `retrieval`: Time taken to retrieve relevant passages from the vector store.
    -   `generation`: Time taken by the LLM to generate an answer based on the prompt.
-   **`baseline_quality_metrics.json`**: Contains metrics related to the quality and correctness of the pipeline's output. Includes:
    -   `retrieval`: Scores like NDCG@10, Precision@10, Recall@10.
    -   `generation_rouge`: ROUGE scores (rouge1, rouge2, rougeL) for answerable questions.
    -   `unanswerable_accuracy`: Accuracy in correctly identifying unanswerable questions.
    -   `ragas`: End-to-end RAG quality scores like Faithfulness and Answer Relevancy.
-   **`frozen/`**: May contain copies of results files from specific, finalized runs (e.g., the official baseline run for the thesis) to preserve them from being overwritten by subsequent runs.

These files provide the quantitative data used to assess the performance of the baseline RAG system.