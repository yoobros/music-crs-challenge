# Technical References

This project's final pipeline is `ensemble__bm25_qmr-qemb_twotower_8b__gbm`.
The papers and technical reports below informed the implementation. The listed
components use project-specific query/document formatting, feature engineering,
and validation logic.

| Component | Technique | Reference | Notes |
| --- | --- | --- | --- |
| `bm25` | Okapi BM25 sparse retrieval | Robertson et al., ["Okapi at TREC-3"](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/okapi_trec3.pdf), TREC 1994 | Used via `bm25s`; query/corpus text composition is project-specific. |
| `bm25_qmr`, ensemble fusion | Reciprocal Rank Fusion | Cormack et al., ["Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods"](https://doi.org/10.1145/1571941.1572114), SIGIR 2009 | Used for BM25+qemb fusion and fold/pool bagging. |
| `qemb_metadata_rich`, `qemb_twotower_8b` backbone | Qwen3 Embedding | Zhang et al., ["Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models"](https://arxiv.org/abs/2506.05176), 2025 | Uses Qwen3-Embedding-0.6B/8B with project-specific query/doc formatting. |
| Two-tower training | Two-tower retrieval, in-batch negatives, logQ correction | Yi et al., ["Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations"](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/), KDD 2019 | Closest reference for two-tower training with in-batch negative bias correction. |
| Two-tower loss | InfoNCE / supervised contrastive, multi-positive variant | van den Oord et al., ["Representation Learning with Contrastive Predictive Coding"](https://arxiv.org/abs/1807.03748), 2018; Khosla et al., ["Supervised Contrastive Learning"](https://arxiv.org/abs/2004.11362), NeurIPS 2020 | Implementation uses multi-positive InfoNCE over query-positive groups. |
| QLoRA adapter training | LoRA / QLoRA | Hu et al., ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685), 2021; Dettmers et al., ["QLoRA: Efficient Finetuning of Quantized LLMs"](https://arxiv.org/abs/2305.14314), NeurIPS 2023 | The 8B embedding model is adapted with 4-bit QLoRA. |
| `gbm` reranker | LambdaRank / LambdaMART-style learning-to-rank with LightGBM | Burges, ["From RankNet to LambdaRank to LambdaMART: An Overview"](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf), 2010; Ke et al., ["LightGBM: A Highly Efficient Gradient Boosting Decision Tree"](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree), NeurIPS 2017 | LightGBM `lambdarank` over leakage-free, fold-aware OOF retrieval features. |
| PAS response | Propose-Assign-Select-style decomposition + DSPy few-shot program | Wang et al., ["Goal-Driven Explainable Clustering via Language Descriptions"](https://aclanthology.org/2023.emnlp-main.657/), EMNLP 2023; Khattab et al., ["DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines"](https://arxiv.org/abs/2310.03714), 2023 | Adapted, not copied: this repo applies a PAS-like decomposition to grounded CRS response generation. |
| Response evaluation framing | LLM-as-a-Judge | Zheng et al., ["Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"](https://arxiv.org/abs/2306.05685), NeurIPS 2023; Liu et al., ["G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"](https://arxiv.org/abs/2303.16634), EMNLP 2023 | Relevant for rubric/prompting style, not core retrieval. |
