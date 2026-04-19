# BERT Classification and Adversarial Robustness Report

## 1. Executive Summary
This report details the implementation, training, and robustness evaluation of a BERT-based sequence classification model. The project aimed to develop a high-accuracy classifier using `bert-base-uncased` and subsequently evaluate its resilience against various adversarial attack vectors, including linguistic permutations, noise injection, and style transfer. 

The final model achieved exceptional performance on the clean test set (AUC 0.99); however, adversarial testing revealed a significant vulnerability to noise injection, despite maintaining robustness against paraphrasing and style variations.

---

## 2. Model Training Methodology
The model was built using the Hugging Face `transformers` library, specifically the `BertForSequenceClassification` architecture. 

### 2.1 Training Strategy: Three-Phase Fine-Tuning
To ensure stable convergence and prevent catastrophic forgetting, a progressive unfreezing strategy was implemented:

| Phase | Strategy | Learning Rate | Goal |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Freeze BERT Base | $1 \times 10^{-3}$ | Train the classification head only to stabilize initial weights. |
| **Phase 2** | Unfreeze Last 2 Layers | $1 \times 10^{-5}$ | Allow high-level feature extraction layers to adapt to the specific domain. |
| **Phase 3** | Unfreeze Last 4 Layers | $5 \times 10^{-6}$ | Fine-tune deeper representations for maximum specific accuracy. |

### 2.2 Hyperparameters & Controls
- **Optimizer**: AdamW
- **Early Stopping**: Patience of 5 epochs monitored on validation loss.
- **Precision**: FP16 mixed precision for efficient GPU utilization.
- **Data Split**: 70% Training, 15% Validation, 15% Testing.

---

## 3. Baseline Performance Results
The model was evaluated on the 15% held-out test set from `final_merged.csv`.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 0.9940 |
| **F1-Score** | 0.9912 |
| **ROC-AUC** | 0.9967 |

> [!NOTE]
> The confusion matrix indicated near-perfect separation, with minimal false positives or false negatives on the baseline test set.

---

## 4. Adversarial Robustness Testing
Robustness was evaluated using an automated pipeline that generated 1,000+ adversarial samples for each attack vector.

### 4.1 Attack Methodologies
1.  **Back-translation**: English $\rightarrow$ German $\rightarrow$ English to introduce natural lexical variation.
2.  **Paraphrasing**: Utilizing `T5-Paraphrase-Paws` to generate semantically equivalent but structurally different inputs.
3.  **Noise Injection**: Character-level perturbations (omissions, swaps, and insertions) to simulate "typos" or low-quality text.
4.  **Style Transfer**: Using `Flan-T5` to rewrite inputs as "Casual/Human" or "AI-generated" styles.

### 4.2 Adversarial Metrics Table

| Attack Vector | Accuracy | F1 (Class 1) | AUC | Robustness Status |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Clean)** | 0.99 | 0.99 | 0.99 | - |
| **Paraphrase 1** | 0.99 | 0.98 | 0.98 | **High** |
| **Paraphrase 2** | 0.98 | 0.96 | 0.97 | **High** |
| **Style Transfer** | 0.97 | 0.92 | 0.95 | **Moderate** |
| **Noise Injection** | 0.93 | 0.76 | 0.81 | **Low** |

---

## 5. Analysis & Discussion
### 5.1 Linguistic Resilience
The model demonstrated high resilience to **Paraphrasing** and **Style Transfer**. This suggests that BERT's attention mechanism successfully identifies the underlying semantic intent even when the surface-level syntax is significantly altered.

### 5.2 The "Noise" Vulnerability
The most significant degradation occurred during **Noise Injection**, where the AUC dropped to **0.81**. 
- **Finding**: Character-level noise (simulating human error) disrupts tokenization, leading to "Unknown" tokens or incorrect embeddings.
- **Impact**: The F1-score for the minority class dropped from 0.99 to 0.76, indicating the model often defaults to the majority class when faced with noisy input.

---

## 6. Conclusion & Recommendations
The developed BERT classifier is highly effective for clean, well-structured text data. However, for deployment in environments where input quality is unpredictable (e.g., social media, live chat), the following are recommended:
1.  **Data Augmentation**: Re-training with noise-augmented samples to improve character-level robustness.
2.  **Preprocessing Layer**: Implementation of a spellchecker or text-normalizer before the BERT encoder.
3.  **Threshold Optimization**: Adjusting classification thresholds specifically for noisy environments to improve the F1-score of the minority class.
