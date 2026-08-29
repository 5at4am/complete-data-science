# BUILD STATUS

> **This file is the single source of truth for what has been built, verified, and what remains.**

Last updated: 2026-08-29

---

## Overall Status

| Metric | Status |
|--------|--------|
| Project scaffolding | ✅ COMPLETE |
| Recovery infrastructure | ✅ COMPLETE |
| Roadmap design | ✅ COMPLETE |
| Notebook templates | ✅ COMPLETE |
| Tracking systems | ✅ COMPLETE |
| Reference docs | ✅ COMPLETE |
| Environment verification | ✅ COMPLETE |
| Phase 00 (Environment) | ✅ COMPLETE |
| Phase 01 (Python) | ✅ COMPLETE |
| Phase 02 (Math) | ✅ COMPLETE |
| Phase 03 (Statistics) | ✅ COMPLETE |
| Phase 04 (Data Analysis) | ✅ COMPLETE |
| Phase 05 (Machine Learning) | ⏳ IN PROGRESS |
| Phase 06 (Deep Learning) | ✅ COMPLETE |
| Phase 07 (NLP) | ✅ COMPLETE |
| Phase 08 (Transformers) | ✅ COMPLETE |
| Phase 09 (Generative AI) | ✅ COMPLETE |
| Phase 10 (LLMs) | ⏳ NOT STARTED |
| Phase 11 (RAG) | ✅ COMPLETE |
| Phase 12 (LangChain) | ✅ COMPLETE |
| Phase 13 (LangGraph) | ✅ COMPLETE |
| Phase 14 (Agents) | ⏳ IN PROGRESS |
| Phase 15 (Evaluation) | ✅ COMPLETE |
| Phase 16 (Deployment) | ✅ COMPLETE |
| Phase 17 (Capstone) | ✅ COMPLETE |

---

## Verification Status Legend

- ✅ **VERIFIED** — Executed from clean kernel, all cells pass, outputs correct.
- ⏳ **IN PROGRESS** — Currently being built or verified.
- ⚠️ **PARTIAL** — Some cells pass, some fail or blocked.
- ❌ **FAILED** — Execution failed, needs fixes.
- 🚫 **BLOCKED** — Cannot execute (missing package, no GPU, no internet, missing dataset, missing API key).

---

## Phase Status Detail

### Phase 00 — Environment Setup

| Unit | Notebook | Status | Last Verified | Notes |
|------|----------|--------|---------------|-------|
| 00.1 | `00_environment/00_environment_verification.ipynb` | ✅ VERIFIED | 2026-08-28 | All core packages OK; opendatasets blocked (Python 3.13 cgi removal) |
| 00.2 | `00_environment/00_02_jupyter_workflow.ipynb` | ✅ VERIFIED | 2026-08-28 | Jupyter cells, kernels, magic |
| 00.3 | `00_environment/00_03_git_version_control.ipynb` | ✅ VERIFIED | 2026-08-28 | Git basics, branching, recovery |
| 00.4 | `00_environment/00_04_python_environments.ipynb` | ✅ VERIFIED | 2026-08-28 | venv, uv, packages |

---

### Phase 01 — Python Foundations

| Unit | Notebook | Status | Last Verified | Notes |
|------|----------|--------|---------------|-------|
| 01.1 | `01_python/01_01_python_basics.ipynb` | ✅ VERIFIED | 2026-08-28 | Variables, types, operators, control flow |
| 01.2 | `01_python/01_02_data_structures.ipynb` | ✅ VERIFIED | 2026-08-28 | Lists, dicts, sets, tuples |
| 01.3 | `01_python/01_03_functions_scope.ipynb` | ✅ VERIFIED | 2026-08-28 | Functions, scope, closures |
| 01.4 | `01_python/01_04_oop.ipynb` | ✅ VERIFIED | 2026-08-28 | Classes, inheritance, dunder |
| 01.5 | `01_python/01_05_numpy.ipynb` | ✅ VERIFIED | 2026-08-28 | NumPy arrays, broadcasting |
| 01.6 | `01_python/01_06_pandas.ipynb` | ✅ VERIFIED | 2026-08-28 | DataFrames, Series |
| 01.7 | `01_python/01_07_matplotlib.ipynb` | ✅ VERIFIED | 2026-08-28 | Plotting |
| 01.8 | `01_python/01_08_file_io_errors.ipynb` | ✅ VERIFIED | 2026-08-28 | File I/O, exceptions |
| 01.9 | `01_python/01_09_synthesis.ipynb` | ✅ VERIFIED | 2026-08-28 | Mini project |

---

### Phase 02 — Mathematics for ML

| Unit | Notebook | Status | Last Verified | Notes |
|------|----------|--------|---------------|-------|
| 02.1 | `02_math/02_01_vectors_matrices.ipynb` | ✅ VERIFIED | 2026-08-28 | Vectors, matrices, norms, dot product |
| 02.2 | `02_math/02_02_matrix_operations.ipynb` | ✅ VERIFIED | 2026-08-28 | Matrix multiplication, inverse |
| 02.3 | `02_math/02_03_derivatives_gradients.ipynb` | ✅ VERIFIED | 2026-08-28 | Derivatives, gradients |
| 02.4 | `02_math/02_04_gradient_descent.ipynb` | ✅ VERIFIED | 2026-08-28 | Gradient descent |
| 02.5 | `02_math/02_05_probability.ipynb` | ✅ VERIFIED | 2026-08-28 | Probability |
| 02.6 | `02_math/02_06_information_theory.ipynb` | ✅ VERIFIED | 2026-08-28 | Information theory |
| 02.7 | `02_math/02_07_eigenvalues.ipynb` | ✅ VERIFIED | 2026-08-28 | Eigenvalues |
| 02.8 | `02_math/02_08_synthesis.ipynb` | ✅ VERIFIED | 2026-08-28 | Synthesis |

---

### Phase 03 — Statistics & Probability

| Unit | Notebook | Status | Last Verified | Notes |
|------|----------|--------|---------------|-------|
| 03.1 | `03_statistics/03_01_descriptive_statistics.ipynb` | ✅ VERIFIED | 2026-08-28 | Descriptive statistics |
| 03.2 | `03_statistics/03_02_probability_distributions.ipynb` | ✅ VERIFIED | 2026-08-28 | Probability distributions |
| 03.3 | `03_statistics/03_03_inferential_statistics.ipynb` | ✅ VERIFIED | 2026-08-28 | Inferential statistics & sampling |
| 03.4 | `03_statistics/03_04_hypothesis_testing.ipynb` | ✅ VERIFIED | 2026-08-28 | Hypothesis testing |
| 03.5 | `03_statistics/03_05_correlation_regression.ipynb` | ✅ VERIFIED | 2026-08-28 | Correlation & regression basics |
| 03.6 | `03_statistics/03_06_bayesian_thinking.ipynb` | ✅ VERIFIED | 2026-08-28 | Bayesian thinking |
| 03.7 | `03_statistics/03_07_statistics_for_ml.ipynb` | ✅ VERIFIED | 2026-08-28 | Statistics for ML |
| 03.8 | `03_statistics/03_08_synthesis.ipynb` | ✅ VERIFIED | 2026-08-28 | Synthesis |

### Phase 04 — Data Analysis & Preparation

| Unit | Notebook | Status | Last Verified | Notes |
|------|----------|--------|---------------|-------|
| 04.1 | `04_data_analysis/04_01_eda.ipynb` | ✅ VERIFIED | 2026-08-28 | Data exploration & EDA |
| 04.2 | `04_data_analysis/04_02_data_cleaning.ipynb` | ✅ VERIFIED | 2026-08-28 | Data cleaning |
| 04.3 | `04_data_analysis/04_03_missing_values.ipynb` | ✅ VERIFIED | 2026-08-28 | Missing values |
| 04.4 | `04_data_analysis/04_04_outliers.ipynb` | ✅ VERIFIED | 2026-08-28 | Outliers |
| 04.5 | `04_data_analysis/04_05_feature_scaling_encoding.ipynb` | ✅ VERIFIED | 2026-08-28 | Feature scaling & encoding |
| 04.6 | `04_data_analysis/04_06_train_val_test_splits.ipynb` | ✅ VERIFIED | 2026-08-28 | Train/val/test splits |
| 04.7 | `04_data_analysis/04_07_data_leakage.ipynb` | ✅ VERIFIED | 2026-08-28 | Data leakage |
| 04.8 | `04_data_analysis/04_08_synthesis.ipynb` | ✅ VERIFIED | 2026-08-28 | Data preparation synthesis |

### Phase 05 — Machine Learning

| Unit | Notebook | Status | Last Verified | Notes |
|------|----------|--------|---------------|-------|
| 05.1 | `05_machine_learning/05_01_ml_fundamentals.ipynb` | ✅ VERIFIED | 2026-08-28 | ML fundamentals & problem formulation |
| 05.2 | `05_machine_learning/05_02_linear_regression.ipynb` | ✅ VERIFIED | 2026-08-28 | Linear regression from scratch |
| 05.3 | `05_machine_learning/05_03_logistic_regression.ipynb` | ✅ VERIFIED | 2026-08-28 | Logistic regression from scratch |
| 05.4 | `05_machine_learning/05_04_model_evaluation.ipynb` | ✅ VERIFIED | 2026-08-28 | Model evaluation |
| 05.5 | `05_machine_learning/05_05_decision_trees.ipynb` | ✅ VERIFIED | 2026-08-28 | Decision trees |
| 05.6 | `05_machine_learning/05_06_random_forests.ipynb` | ✅ VERIFIED | 2026-08-28 | Random forests |
| 05.7 | `05_machine_learning/05_07_gradient_boosting.ipynb` | ✅ VERIFIED | 2026-08-28 | Gradient boosting (XGBoost, LightGBM) |
| 05.8 | `05_machine_learning/05_08_knn.ipynb` | ⏳ IN PROGRESS | — | k-Nearest Neighbors |
| 05.9 | `05_machine_learning/05_09_naive_bayes.ipynb` | ⏳ IN PROGRESS | — | Naive Bayes |
| 05.10 | `05_machine_learning/05_10_svm.ipynb` | ⏳ IN PROGRESS | — | Support Vector Machines |
| 05.11 | `05_machine_learning/05_11_clustering.ipynb` | ⏳ IN PROGRESS | — | Clustering |
| 05.12 | `05_machine_learning/05_12_pca.ipynb` | ⏳ IN PROGRESS | — | Dimensionality Reduction (PCA) |
| 05.13 | `05_machine_learning/05_13_feature_engineering.ipynb` | ⏳ IN PROGRESS | — | Feature Engineering |
| 05.14 | `05_machine_learning/05_14_imbalanced_learning.ipynb` | ⏳ IN PROGRESS | — | Imbalanced Learning |
| 05.15 | `05_machine_learning/05_15_cv_hyperparameter_tuning.ipynb` | ⏳ IN PROGRESS | — | Cross-Validation & Tuning |
| 05.16 | `05_machine_learning/05_16_model_interpretation.ipynb` | ⏳ IN PROGRESS | — | Model Interpretation |
| 05.17 | `05_machine_learning/05_17_ensembling.ipynb` | ⏳ IN PROGRESS | — | Ensembling |
| 05.18 | `05_machine_learning/05_18_synthesis.ipynb` | ⏳ IN PROGRESS | — | ML Synthesis & Review |

### Phase 06 — Deep Learning

| Unit | Notebook | Status | Last Verified | Notes |
|------|----------|--------|---------------|-------|
| 06.1 | `06_deep_learning/06_01_perceptron_activation.ipynb` | ✅ VERIFIED | 2026-08-29 | Perceptron & activation functions |
| 06.2 | `06_deep_learning/06_02_loss_functions.ipynb` | ✅ VERIFIED | 2026-08-29 | Loss functions |
| 06.3 | `06_deep_learning/06_03_backpropagation.ipynb` | ✅ VERIFIED | 2026-08-29 | Gradient descent & backprop from scratch |
| 06.4 | `06_deep_learning/06_04_mlp_pytorch.ipynb` | ✅ VERIFIED | 2026-08-29 | MLP with PyTorch |
| 06.5 | `06_deep_learning/06_05_regularization.ipynb` | ✅ VERIFIED | 2026-08-29 | Regularization |
| 06.6 | `06_deep_learning/06_06_optimizers_lr_schedules.ipynb` | ✅ VERIFIED | 2026-08-29 | Optimizers & LR schedules |
| 06.7 | `06_deep_learning/06_07_datasets_dataloaders.ipynb` | ✅ VERIFIED | 2026-08-29 | PyTorch Datasets & DataLoaders |
| 06.8 | `06_deep_learning/06_08_training_loops_validation.ipynb` | ✅ VERIFIED | 2026-08-29 | Training loops & validation |
| 06.9 | `06_deep_learning/06_09_checkpointing_transfer_learning.ipynb` | ✅ VERIFIED | 2026-08-29 | Checkpointing & transfer learning |
| 06.10 | `06_deep_learning/06_10_cnns.ipynb` | ✅ VERIFIED | 2026-08-29 | CNNs for computer vision |
| 06.11 | `06_deep_learning/06_11_rnn_lstm_gru.ipynb` | ✅ VERIFIED | 2026-08-29 | RNNs, LSTMs, GRUs |
| 06.12 | `06_deep_learning/06_12_attention.ipynb` | ✅ VERIFIED | 2026-08-29 | Attention mechanism |
| 06.13 | `06_deep_learning/06_13_synthesis.ipynb` | ✅ VERIFIED | 2026-08-29 | Deep learning synthesis & review |

### Phase 07 — NLP

| Unit | Notebook | Status | Last Verified | Notes |
|------|----------|--------|---------------|-------|
| 07.1 | `07_nlp/07_01_text_preprocessing.ipynb` | ✅ VERIFIED | 2026-08-29 | Text preprocessing & tokenization |
| 07.2 | `07_nlp/07_02_bow_tfidf.ipynb` | ✅ VERIFIED | 2026-08-29 | Bag of words & TF-IDF |
| 07.3 | `07_nlp/07_03_ngrams.ipynb` | ✅ VERIFIED | 2026-08-29 | N-grams |
| 07.4 | `07_nlp/07_04_text_classification.ipynb` | ✅ VERIFIED | 2026-08-29 | Text classification |
| 07.5 | `07_nlp/07_05_word_embeddings.ipynb` | ✅ VERIFIED | 2026-08-29 | Word embeddings (Word2Vec concepts) |
| 07.6 | `07_nlp/07_06_sequence_models.ipynb` | ✅ VERIFIED | 2026-08-29 | Sequence models for NLP |
| 07.7 | `07_nlp/07_07_attention_nlp.ipynb` | ✅ VERIFIED | 2026-08-29 | Attention for NLP |
| 07.8 | `07_nlp/07_08_nlp_evaluation.ipynb` | ✅ VERIFIED | 2026-08-29 | NLP evaluation |
| 07.9 | `07_nlp/07_09_synthesis.ipynb` | ✅ VERIFIED | 2026-08-29 | NLP synthesis & review |

### Phase 08 — Transformers

| Unit | Notebook | Status | Last Verified | Notes |
|------|----------|--------|---------------|-------|
| 08.1 | `08_transformers/08_01_transformer_architecture.ipynb` | ✅ VERIFIED | 2026-08-29 | Transformer architecture from scratch |
| 08.2 | `08_transformers/08_02_self_attention.ipynb` | ✅ VERIFIED | 2026-08-29 | Self-attention deep dive |
| 08.3 | `08_transformers/08_03_positional_encoding.ipynb` | ✅ VERIFIED | 2026-08-29 | Positional encoding |
| 08.4 | `08_transformers/08_04_encoder_decoder.ipynb` | ✅ VERIFIED | 2026-08-29 | Encoder-decoder architecture |
| 08.5 | `08_transformers/08_05_bert_style.ipynb` | ✅ VERIFIED | 2026-08-29 | BERT-style models |
| 08.6 | `08_transformers/08_06_gpt_style.ipynb` | ✅ VERIFIED | 2026-08-29 | Causal language models (GPT-style) |
| 08.7 | `08_transformers/08_07_tokenizers.ipynb` | ✅ VERIFIED | 2026-08-29 | Tokenizers (BPE, WordPiece) |
| 08.8 | `08_transformers/08_08_huggingface_ecosystem.ipynb` | ✅ VERIFIED | 2026-08-29 | Hugging Face ecosystem |
| 08.9 | `08_transformers/08_09_finetuning_transformers.ipynb` | ✅ VERIFIED | 2026-08-29 | Fine-tuning transformers |
| 08.10 | `08_transformers/08_10_synthesis.ipynb` | ✅ VERIFIED | 2026-08-29 | Transformers synthesis & review |

### Phase 09 — Generative AI

| Unit | Notebook | Status | Last Verified | Notes |
|------|----------|--------|---------------|-------|
| 09.1 | `09_generative_ai/09_01_what_is_llm.ipynb` | ✅ VERIFIED | 2026-08-29 | What is a language model |
| 09.2 | `09_generative_ai/09_02_tokens_tokenization.ipynb` | ✅ VERIFIED | 2026-08-29 | Tokens & tokenization |
| 09.3 | `09_generative_ai/09_03_embeddings.ipynb` | ✅ VERIFIED | 2026-08-29 | Embeddings |
| 09.4 | `09_generative_ai/09_04_attention_transformer_recap.ipynb` | ✅ VERIFIED | 2026-08-29 | Attention & transformer recap |
| 09.5 | `09_generative_ai/09_05_pretraining.ipynb` | ✅ VERIFIED | 2026-08-29 | Pretraining |
| 09.6 | `09_generative_ai/09_06_inference_decoding.ipynb` | ✅ VERIFIED | 2026-08-29 | Inference & decoding |
| 09.7 | `09_generative_ai/09_07_temperature_sampling.ipynb` | ✅ VERIFIED | 2026-08-29 | Temperature & sampling |
| 09.8 | `09_generative_ai/09_08_instruction_alignment.ipynb` | ✅ VERIFIED | 2026-08-29 | Instruction following & alignment |
| 09.9 | `09_generative_ai/09_09_llm_apis.ipynb` | ✅ VERIFIED | 2026-08-29 | LLM APIs |
| 09.10 | `09_generative_ai/09_10_prompt_engineering.ipynb` | ✅ VERIFIED | 2026-08-29 | Prompt engineering |
| 09.11 | `09_generative_ai/09_11_structured_output.ipynb` | ✅ VERIFIED | 2026-08-29 | Structured output |
| 09.12 | `09_generative_ai/09_12_function_tool_calling.ipynb` | ✅ VERIFIED | 2026-08-29 | Function/tool calling |
| 09.13 | `09_generative_ai/09_13_streaming_caching_retries.ipynb` | ✅ VERIFIED | 2026-08-29 | Streaming, caching, retries |
| 09.14 | `09_generative_ai/09_14_multimodal_overview.ipynb` | ✅ VERIFIED | 2026-08-29 | Multimodal generative AI overview |
| 09.15 | `09_generative_ai/09_15_synthesis.ipynb` | ✅ VERIFIED | 2026-08-29 | Generative AI synthesis & review |

---

## Environment Snapshot

| Component | Version | Status |
|-----------|---------|--------|
| Python | 3.13.9 (uv venv) | ✅ |
| uv | 0.9.9 | ✅ |
| Git | initialized (no commits yet) | ✅ |
| GPU | None (CPU-only) | ⚠️ CPU-only |
| Internet | Available | ✅ |
| Jupyter | 4.6.3 (jupyterlab) | ✅ |
| numpy | 2.5.2 | ✅ |
| pandas | 3.0.5 | ✅ |
| scikit-learn | 1.9.0 | ✅ |
| scipy | 1.18.1 | ✅ |
| matplotlib | 3.11.1 | ✅ |
| seaborn | 0.13.2 | ✅ |
| torch | 2.13.0 (CPU) | ✅ |
| transformers | 5.16.1 | ✅ |
| datasets | 5.0.1 | ✅ |
| langchain | 1.3.18 | ✅ |
| langgraph | installed | ✅ |
| xgboost | 3.4.1 | ✅ |
| lightgbm | 4.7.0 | ✅ |
| chromadb | 1.5.9 | ✅ |
| faiss-cpu | 1.15.0 | ✅ |
| fastapi | 0.141.1 | ✅ |
| opendatasets | 0.1.22 | 🚫 BLOCKED (Python 3.13 removed `cgi`) |

---

## Blocked Items

| Item | Reason |
|------|--------|
| opendatasets | Python 3.13 removed the `cgi` module which opendatasets depends on. Use `kagglehub` instead for dataset downloads. |

---

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-08-28 | Use uv-managed virtual environment | User's chosen environment manager |
| 2026-08-28 | CPU-only PyTorch | No GPU detected on this machine |
| 2026-08-28 | Build docs/architecture/roadmap first, notebooks after | Per master build instruction §51 |
