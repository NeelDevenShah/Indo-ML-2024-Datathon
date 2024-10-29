Below is an extended and detailed `README.md` file that goes in-depth into each model and methodological component of the project, as well as the experimental setup, results, and future directions.

---

# Attribute-Value Prediction From E-Commerce Product Descriptions

## Project Overview

The goal of this project is to automate attribute-value prediction in e-commerce catalogs by extracting and predicting hierarchical attributes from unstructured product descriptions. Automated attribute prediction enhances e-commerce platforms by improving search relevance, recommendation systems, and overall customer experience. The project explores five advanced approaches, including hierarchical classification and agent-based models, each designed to address the complex, multi-level classification demands of e-commerce.

### Team and Attribution

- **Authors**: Neel Shah, Sneh Shah
- **Institution**: Charotar University of Science and Technology
- **Contributors**: Equal contributions from both authors.

---

## Table of Contents

- [Key Contributions](#key-contributions)
- [Model Architectures and Methodologies](#model-architectures-and-methodologies)
  - [Multi-Learning Hierarchical Transformer (MLHT)](#multi-learning-hierarchical-transformer-mlht)
  - [Expert-Ensemble Multi-Learning Hierarchical Transformer (EE-MLHT)](#expert-ensemble-multi-learning-hierarchical-transformer-ee-mlht)
  - [Selective Pathway Transformer Classifier (SPTC)](#selective-pathway-transformer-classifier-sptc)
  - [Agent-Based Models](#agent-based-models)
  - [Semantic Search-Driven Agentic Classification (SSDAC)](#semantic-search-driven-agentic-classification-ssdac)
- [Training Setup](#training-setup)
- [Results](#results)
- [Experiments and Insights](#experiments-and-insights)
- [Challenges and Future Directions](#challenges-and-future-directions)
- [Acknowledgments](#acknowledgments)

---

## Key Contributions

The project presents five distinct methodologies, each addressing specific aspects of hierarchical e-commerce classification:

1. **Multi-Learning Hierarchical Transformer (MLHT)**: Incorporates supervised, reinforcement, and contrastive learning to model multi-level relationships between attribute categories.
2. **Expert-Ensemble Multi-Learning Hierarchical Transformer (EE-MLHT)**: Addresses class imbalance and knowledge transfer through an ensemble of specialized expert models.
3. **Selective Pathway Transformer Classifier (SPTC)**: Optimizes classification paths by employing a hierarchical pathway strategy.
4. **Agentic Approaches**: Designed to simulate human reasoning in hierarchical classification, including the Dynamic Description-Driven Agentic Classification (DDAAC), Smart Search-Driven Agentic Classification (SSDAC), and Hierarchical Option-Based Agentic Classification (HOAC).
5. **Semantic Search-Driven Agentic Classification (SSDAC)**: Applies semantic similarity to enhance attribute prediction accuracy through vector-based searches.

---

## Model Architectures and Methodologies

### Multi-Learning Hierarchical Transformer (MLHT)

**Objective**: The MLHT model addresses the hierarchical complexity in e-commerce product categorization by classifying across four levels (Supergroup, Group, Module, Brand).

- **Architecture**:
  - **Backbones**: Two variants of the MLHT were tested using DeBERTa and XLM-RoBERTa.
  - **Hierarchical Classifiers**: Four-level classifier with fully connected layers. Each level takes in the features from previous levels, forming a cascading dependency that preserves hierarchical information.
- **Learning Techniques**:
  - **Reinforcement Learning Policies**: Probabilistic decision-making through policy networks allows the model to handle uncertainties in classification.
  - **Contrastive Learning**: Applied at each hierarchical level to enhance separation between similar classes.
  - **Few-Shot Learning**: Uses prototype-based learning for classes with minimal data to avoid overfitting.
  - **Loss Functions**: The multi-objective loss function includes cross-entropy for supervised learning, contrastive loss for class separation, and reinforcement learning rewards.

### Expert-Ensemble Multi-Learning Hierarchical Transformer (EE-MLHT)

**Objective**: The EE-MLHT model focuses on handling class imbalance through a structured ensemble of expert classifiers at each hierarchical level.

- **Expert Classifiers**:

  - **Role-Specific Classifiers**: Each classifier in the ensemble is tuned for a specific level in the hierarchy (Supergroup, Group, Module, Brand).
  - **Knowledge Distillation**: Aligns the outputs of expert classifiers with those of the main classifier by softening predictions, improving generalization.
  - **Confidence-Weighted Loss**: Incorporates confidence-weighted loss to prioritize predictions with higher confidence scores.

- **Loss Components**:
  - **Focal Loss**: Emphasizes harder-to-classify examples to reduce class imbalance effects.
  - **Weighted Ensemble**: Confidence-based weighting ensures reliance on predictions with higher reliability.

### Selective Pathway Transformer Classifier (SPTC)

**Objective**: SPTC was developed to improve classification accuracy by enforcing constraints based on hierarchical mappings, making each level aware of allowable classes.

- **Model Design**:
  - **Backbone**: BERT-of-Theseus, a more efficient version of BERT, allows reduced computational cost without sacrificing performance.
  - **Hierarchical Mapping**: Each classifier head receives constraints based on valid mappings from the preceding level, which limits its prediction space to the current level’s allowable categories.
- **Pathway Loss**: Custom loss function penalizes predictions falling outside valid category ranges, enhancing accuracy in multi-class classification by reducing interference from irrelevant categories.

### Agent-Based Models

#### Dynamic Description-Driven Agentic Classification (DDAAC)

- **Description Generation Agent**: Uses Llama3.1 to expand abbreviated labels into descriptive narratives for better interpretability.
- **Classification Agents**: Supergroup, Group, Module, and Brand classifiers process detailed descriptions, achieving granular classification accuracy through iterative refinement.

#### Smart Search-Driven Agentic Classification (SSDAC)

- **Google API Agent**: Retrieves detailed product descriptions using Google’s search engine for highly accurate, pre-existing descriptions. This avoids the limitations of generative language models for unseen product labels.
- **Hierarchical Classification**: Uses the same hierarchical agents as DDAAC, benefiting from Google’s ranking algorithms to prioritize authoritative sources.

#### Hierarchical Option-Based Agentic Classification (HOAC)

- **Hierarchical Filtering**: Successive classification stages (Supergroup, Group, Module, Brand) filter options based on prior predictions, focusing only on relevant subclasses.
- **Evaluation of Configurations**: Experimented with different hierarchies—Supergroup-first and Brand-first—and found that a Supergroup-based hierarchy offered clearer distinction for product types.

### Semantic Search-Driven Agentic Classification (SSDAC)

**Objective**: Enhance classification accuracy by leveraging sentence embeddings for semantic similarity. This approach is effective in distinguishing between categories with close semantic overlap.

- **Architecture**:
  - **Sentence-Transformers/all-MiniLM-L6-v2**: Used for generating embeddings that encode the semantic content of descriptions. This model is optimized for real-time applications with efficient processing.
  - **Semantic Search**: Queries are matched with indexed descriptions based on high-dimensional vector similarities, improving classification accuracy for closely related classes.

---

## Training Setup

The models were trained on **Kaggle P100 GPUs** with a training strategy that utilized 12-hour sessions to maximize GPU usage within Kaggle’s weekly limit. Each session involved saving model checkpoints for continuous learning over multiple iterations.

- **Frameworks**:
  - **Hugging Face Transformers** and **PyTorch** for model implementation.
  - **txtai** for semantic indexing and search in SSDAC.
- **Compute Requirements**:
  - **GPU**: NVIDIA Tesla P100 (15GB GPU RAM).
  - **RAM**: 30GB CPU RAM for batch processing and model optimization.

This training methodology provided flexibility in resource management while allowing continuous monitoring of model performance.

---

## Results

### Performance Metrics

The table below summarizes the performance of each model:

| Model                | Item Accuracy | CPU Time (ms) | CUDA Time (ms) | GPU FLOPS (Train) | Inference Time (s) | GPU FLOPS (Infer) |
| -------------------- | ------------- | ------------- | -------------- | ----------------- | ------------------ | ----------------- |
| **MLHT (DeBERTa)**   | 0.4087        | 867.567       | 205.878        | 52.0525           | 0.3503             | 2.7820            |
| **EE-MLHT**          | 0.3990        | 644.858       | 350.747        | 412.0799          | 0.6264             | 90.2679           |
| **SPTC**             | 0.3454        | 995.420       | 656.425        | 654.0156          | 0.6291             | 11.9201           |
| **SSDAC (Semantic)** | 0.2875        | NA            | NA             | NA                | 8.6787             | 0.0               |
| **DDAAC**            | 0.2831        | 16.513        | NA             | 1.0218            | 0.0756             | 0.1277            |

### Insights from Results

- **MLHT (DeBERTa)** demonstrated the highest accuracy and inference efficiency, outperforming XLM-RoBERTa in both accuracy and GPU processing time.
- **EE-MLHT** performed well in difficult-to-classify cases thanks to the integration of knowledge distillation and confidence-weighted ensemble strategies.
- **Agent-Based Models** (DDAAC and SSDAC) offered increased flexibility in dealing with varied product descriptions,

though SSDAC was constrained by the rate limits of Google API.

---

## Experiments and Insights

1. **Feature Selection**: Using descriptions alone produced the best results, as additional information such as retailer data reduced accuracy, introducing unnecessary noise.
2. **Handling Class Imbalance**: Knowledge distillation and focal loss in EE-MLHT proved effective for addressing the high imbalance across e-commerce product categories.
3. **Scalability of Agent-Based Approaches**: While the agent-based methods (especially SSDAC) improved interpretability, limitations in scaling up due to rate limiting and computational demands highlighted challenges in real-time application.
4. **Hierarchical Pathways in SPTC**: The SPTC’s hierarchical mappings proved critical for consistent predictions across levels, though these mappings may require reconfiguration for different domains.

---

## Challenges and Future Directions

### Challenges

1. **Scalability Constraints**: Rate limits on Google API and computational constraints on larger agent-based models limited the agentic approaches’ applicability for larger datasets.
2. **Cascading Errors in Hierarchical Filtering**: Errors in higher levels (e.g., Supergroup misclassification) significantly impacted downstream predictions, suggesting a need for error mitigation in hierarchical pathways.

### Future Directions

1. **Larger Language Models**: Access to models with 40 billion parameters or more could further boost performance, especially in agentic tasks where knowledge depth is beneficial.
2. **Advanced Error-Handling Mechanisms**: Implementing early-error correction or real-time feedback in hierarchical pathways could prevent misclassification propagation, improving accuracy at every level.
3. **Real-Time Semantic Search**: Expanding semantic search with real-time indexing could improve model adaptability to new product categories, especially in rapidly evolving e-commerce environments.

---

## Acknowledgments

We extend our gratitude to the IndoML 2024 team for organizing this datathon, Kaggle for providing free GPU resources, and NielsenIQ for sponsoring the event. These resources and support were crucial in allowing for the extensive experimentation and model training necessary to achieve high-performance classification.
