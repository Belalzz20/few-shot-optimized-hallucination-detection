# Few-Shot Optimized Hallucination Detection

A lightweight, deterministic, and black-box framework for detecting hallucinated answers in **knowledge-grounded question answering (QA)** systems.

The detection component verifies whether a candidate answer is supported by a provided knowledge passage and question. It combines a fine-tuned **DeBERTa-v3-base** verifier with optional **question-aware knowledge summarization** to reduce irrelevant context before verification.

> This repository contains the implementation and experiments for the hallucination-detection component of the graduation project **"Few-Shot Optimized Framework for Secure Hallucination Detection in Resource-Limited NLP Systems."**

---

## Overview

Large Language Models (LLMs) can generate fluent and convincing answers that are not supported by the available evidence.

This project formulates hallucination detection as a supervised verification task over three inputs:

* **K** — Knowledge passage
* **Q** — Question
* **A** — Candidate answer

The verifier predicts one of two classes:

* `Supported (1)` — the answer is grounded in the provided knowledge.
* `Hallucinated (0)` — the answer is unsupported or contradicts the provided knowledge.

The main idea is that hallucination detection is not only a classifier problem. Long knowledge passages often contain irrelevant information that can distract the verifier. Therefore, the framework optionally compresses the knowledge passage using a question-aware summarization model before performing verification.

---
## 📄 Research Paper

### Improving Hallucination Detection via Question-Aware Knowledge Summarization

This repository accompanies our research work on improving hallucination detection
in knowledge-grounded question answering through question-aware knowledge
summarization.

The paper has been **accepted for presentation at the 10th International
Conference on Information Technology (InCIT 2026)** and is eligible for
publication in the **IEEE Conference Proceedings**, subject to completion of
the required conference and IEEE publication procedures.

**Authors:**
- Abdelrahman Mohamed
- Belal Hesham
- Ebtesam E. Shemis

📄 **Paper:** [`Improving_Hallucination_Detection.pdf`](./paper/Improving_Hallucination_Detection.pdf)

> **Publication status:** Accepted for presentation at InCIT 2026.
> IEEE Conference Proceedings publication is subject to completion of the
> required publication procedures.

The paper presents the proposed hallucination detection framework, experimental
evaluation, ablation studies, and analysis of question-aware knowledge
summarization for reducing irrelevant context while preserving information
needed for factual verification.
## Key Idea

The verification pipeline can be represented as:

```text
Knowledge (K)
      │
      ▼
Question-Aware Summarization
(Optional)
      │
      ▼
Relevant Knowledge (K̃)
      │
      ├──────────────┐
      │              │
      ▼              ▼
   Question (Q)   Answer (A)
      │              │
      └──────┬───────┘
             ▼
      Input Construction
 [K̃ / K] [SEP] Q [SEP] A
             │
             ▼
      DeBERTa-v3-base
       Fine-Tuned Verifier
             │
             ▼
      ┌───────────────┐
      │               │
      ▼               ▼
  Supported       Hallucinated
    (1)               (0)
```

The original knowledge passage can also be used directly, allowing a controlled comparison between summarized and non-summarized evidence.

---

## Why Question-Aware Summarization?

A knowledge passage may contain many entities, dates, events, and facts that are unrelated to the question being asked.

Instead of simply shortening the input, the framework summarizes the knowledge **conditioned on the question**:

```text
Question: Q
Knowledge: K

        ↓

Question-aware summary: K̃
```

This aims to preserve evidence relevant to the question while removing distracting context.

The experiments compare:

1. Original knowledge
2. Generic summarization
3. Question-aware BART summarization
4. Question-aware PEGASUS summarization
5. Question-aware FLAN-T5 summarization

This setup helps distinguish the effect of **relevance-aware evidence selection** from simple input compression.

---

## Detection Model

### DeBERTa-v3-base

The core verifier is a fine-tuned **DeBERTa-v3-base** transformer encoder.

The model receives:

```text
[CLS] Knowledge [SEP] Question [SEP] Candidate Answer [SEP]
```

with a maximum sequence length of **512 tokens**.

A classification head is applied to the final `[CLS]` representation to predict:

```text
0 → Hallucinated
1 → Supported
```

### Training Configuration

| Parameter               | Value           |
| ----------------------- | --------------- |
| Backbone                | DeBERTa-v3-base |
| Maximum sequence length | 512             |
| Optimizer               | AdamW           |
| Learning rate           | 2 × 10⁻⁵        |
| Weight decay            | 0.01            |
| Warmup ratio            | 0.05            |
| Train batch size        | 8               |
| Evaluation batch size   | 8               |
| Epochs                  | 2               |
| Train/Test split        | 70/30           |
| Hardware                | NVIDIA Tesla T4 |
| Random seed             | 42              |

---

## Summarization Models

The repository evaluates several question-aware summarization backbones:

| Model    | Role                                |
| -------- | ----------------------------------- |
| BART-CNN | Question-aware evidence compression |
| PEGASUS  | Question-aware evidence compression |
| FLAN-T5  | Question-aware evidence compression |

**FLAN-T5** achieved the strongest downstream hallucination-detection performance in the reported experiments.

---

## Datasets

### HaluEval-QA

The primary supervised detection benchmark is **HaluEval-QA**.

Each original QA record is converted into two verification samples:

```text
(K, Q, Correct Answer)      → Supported (1)
(K, Q, Hallucinated Answer) → Hallucinated (0)
```

The project uses a leakage-safe split at the original-record level so that paired supported/hallucinated variants do not cross between training and testing.

The resulting verification split contains:

| Split | Instances | Hallucination | Supported |
| ----- | --------: | ------------: | --------: |
| Train |    14,000 |         7,000 |     7,000 |
| Test  |     6,000 |         3,000 |     3,000 |

---

### HotpotQA-Derived Benchmark

A separate HotpotQA-derived benchmark is used to evaluate generalization under **long-context distribution shift**.

The benchmark contains longer and more reasoning-intensive passages, making it useful for testing whether question-aware evidence compression remains effective when irrelevant context becomes more significant.

---

## Results

### HaluEval-QA

The reported results show that task-specific supervised verification substantially improves over the zero-shot NLI baseline.

| Method                |   Accuracy |  Precision | Recall |         F1 |
| --------------------- | ---------: | ---------: | -----: | ---------: |
| Zero-shot NLI DeBERTa |     0.5280 |     0.6042 | 0.1623 |     0.2559 |
| Fine-tuned DeBERTa    |     0.9770 |     0.9640 | 0.9910 |     0.9773 |
| DeBERTa + FLAN-T5     | **0.9848** | **0.9815** | 0.9883 | **0.9849** |

### Summarization Ablation

| Knowledge Setting          |   Accuracy |         F1 |
| -------------------------- | ---------: | ---------: |
| Original                   |     0.9770 |     0.9773 |
| Generic Summarization      |     0.9767 |     0.9770 |
| Question-Aware BART        |     0.9790 |     0.9793 |
| Question-Aware PEGASUS     |     0.9782 |     0.9785 |
| **Question-Aware FLAN-T5** | **0.9848** | **0.9849** |

These comparisons indicate that compression alone is not sufficient; conditioning the summarization on the question is important for preserving useful verification evidence.

---

## Long-Context Generalization

On the HotpotQA-derived benchmark, question-aware FLAN-T5 summarization achieved:

| Setting                |   Accuracy |   Macro-F1 |
| ---------------------- | ---------: | ---------: |
| Original Knowledge     |     0.5025 |     0.4662 |
| Question-Aware FLAN-T5 | **0.7917** | **0.7905** |

The accompanying research paper reports that a length-matched truncation control reaches only **0.5457 macro-F1**, supporting the interpretation that the improvement is related to selecting question-relevant evidence rather than merely reducing the input length.

---

## Efficiency

Question-aware summarization also reduces the amount of knowledge processed by the verifier.

| Setting    | Avg. Tokens | Latency (ms) | Throughput (samples/s) | GPU Memory |
| ---------- | ----------: | -----------: | ---------------------: | ---------: |
| Original   |      110.95 |        15.99 |                  62.56 |    2.68 GB |
| Summarized |       63.25 |         9.71 |                 103.02 |    1.59 GB |

The summarized setting reduces the average verifier input length while also reducing verifier-side latency and GPU memory usage.

> Note: online summarization introduces an additional preprocessing cost. The summarization stage is therefore most attractive when summaries can be cached or reused.

---

## Summary Faithfulness

Because summarization can potentially remove important evidence, the project evaluates the generated summaries separately.

For the FLAN-T5 configuration:

| Metric                   | Result |
| ------------------------ | -----: |
| Summary Faithfulness     | 92.98% |
| Answer Preservation Rate | 74.78% |
| Compression Ratio        |  0.389 |
| Context Reduction        | 61.05% |

Answer preservation and summary faithfulness are evaluated separately because a summary can remain factually supported by the source while not explicitly containing the original answer span.

---

## Repository Structure

```text
few-shot-optimized-hallucination-detection/
│
├── Experiments/
│   └── Experimental configurations and evaluation artifacts
│
├── generation/
│   └── Answer-generation related components
│
├── hallucination_detector/
│   └── Fine-tuned hallucination verification components
│
├── hallucination_detector_zero/
│   └── Zero-shot / baseline detection components
│
├── original_data/
│   └── Dataset and original knowledge resources
│
├── summarization/
│   └── Question-aware summarization components
│
└── README.md
```

---

## Detection Variants

The framework supports comparing multiple evidence configurations:

```text
1. Original Knowledge
2. BART-Summarized Knowledge
3. FLAN-T5-Summarized Knowledge
4. PEGASUS-Summarized Knowledge
```

This makes it possible to evaluate whether changes in hallucination detection performance are caused by the verifier itself or by the quality and representation of the evidence supplied to it.

---

## Streamlit Interface

The complete project includes a Streamlit-based interface for interactive verification.

The interface allows users to:

* Load a JSON Lines dataset
* Select a sample
* Inspect the knowledge passage and question
* Insert the reference answer
* Insert the hallucinated answer
* Generate a candidate answer
* Select a detector variant
* Run all detector variants in comparison mode
* Display prediction confidence and class probabilities

The detector comparison mode evaluates the same candidate answer using:

```text
Original
BART
FLAN-T5
PEGASUS
```

and reports the predicted class and confidence for each configuration.

---

## Input Format

The dataset uses a JSON Lines (`.jsonl`) format.

A typical record contains:

```json
{
  "knowledge": "...",
  "question": "...",
  "right_answer": "...",
  "hallucinated_answer": "..."
}
```

For verification, the detector operates on:

```text
Knowledge + Question + Candidate Answer
```

---

## Installation

The project is implemented in Python using the Hugging Face Transformers ecosystem.

The main dependencies include:

* Python
* PyTorch
* Hugging Face Transformers
* DeBERTa-v3-base
* FLAN-T5
* BART
* PEGASUS
* Streamlit
* MiniCheck
* AlignScore

For GPU-based experiments, the reported experiments were conducted using an NVIDIA Tesla T4.

> The exact execution command depends on the entry-point script included in the corresponding repository module. The repository structure above intentionally avoids assuming a filename that is not documented by the project.

---

## Experimental Design

The experiments are designed to isolate the effect of evidence representation.

The main comparisons include:

* Original vs. summarized knowledge
* Generic vs. question-aware summarization
* BART vs. PEGASUS vs. FLAN-T5
* Zero-shot NLI vs. task-specific fine-tuning
* DeBERTa vs. BERT, ELECTRA, and RoBERTa
* Supervised verifier vs. external factual-consistency baselines
* Original vs. summarized knowledge under long-context distribution shift

The evaluation uses:

* Accuracy
* Precision
* Recall
* F1
* Macro-F1
* Summary faithfulness
* Answer preservation
* Inference latency
* Throughput
* GPU memory usage

---

## Why This Approach?

The project focuses on a practical hallucination-detection setting:

* **Black-box compatible** — no access to generator hidden states or gradients is required.
* **Deterministic verification** — one stable prediction is produced for each input.
* **Task-specific** — the verifier is trained directly on `(K, Q, A)` verification examples.
* **Evidence-aware** — the framework treats knowledge representation as part of the detection problem.
* **Lightweight** — based on a single fine-tuned encoder rather than an ensemble of multiple judges or stochastic generations.
* **Long-context aware** — question-aware summarization is evaluated specifically under distribution shift.

---

## Limitations

The current system has several limitations:

1. The primary HaluEval-QA evaluation is a controlled and balanced benchmark.
2. The HotpotQA-derived hallucinations are constructed through controlled answer substitution and may not fully represent naturally occurring LLM hallucinations.
3. Summarization can remove subtle evidence or the exact answer span.
4. Online summarization introduces additional preprocessing cost.
5. The current evaluation focuses on English textual QA.
6. The verifier operates within a maximum input length of 512 tokens.
7. The current work does not claim universal robustness against every type of hallucination or deployment environment.

---

## Research Paper

The detection component is described in:

**Improving Hallucination Detection via Question-Aware Knowledge Summarization**

**Authors**

* Abdelrahman Mohamed
* Belal Hesham
* Ebtesam E. Shemis

The paper studies whether question-aware evidence compression can improve hallucination detection by increasing the signal-to-noise ratio of verifier inputs.

---

## Graduation Project

This repository is part of the graduation project:

**Few-Shot Optimized Framework for Secure Hallucination Detection in Resource-Limited NLP Systems**

**Team**

* Abdelrahman Mohamed
* Belal Hesham

**Supervisor**

* Dr. Ebtsam El-Hosseiny

**Institution**

October University for Modern Sciences and Arts (MSA)
Faculty of Computer Science

**Academic Year:** 2025 / 2026

---

## Citation

If you use this implementation or the reported experimental setup, please cite the associated research work:

```text
Abdelrahman Mohamed, Belal Hesham, and Ebtesam E. Shemis.
"Improving Hallucination Detection via Question-Aware Knowledge Summarization."
2026.
```

---

## Acknowledgements

This work builds upon publicly available datasets and pretrained models from the NLP research community, including HaluEval, HotpotQA, DeBERTa-v3, FLAN-T5, BART, and PEGASUS.

---
## 📜 License

Copyright (c) 2026 Belal Hesham, Abdelrahman Mohamed, and Ebtesam E. Shemis.

All rights reserved.

This repository is provided for academic and research reference purposes only.
The source code may not be copied, modified, redistributed, sublicensed,
or incorporated into other projects without prior written permission from
the copyright holders.

Third-party datasets, pretrained models, libraries, and other external
components used in this project are subject to their respective licenses
and terms.
