# AI-ML-Assignment-8-LLM-Evaluation
# Cell 9: Generate README Content
readme_content = f"""# AI/ML Assignment 8: Evaluating Fine-Tuned LLM Performance

**Student Name:** Ermiyas H.  
**Base Model:** DistilBERT (distilbert-base-uncased)  
**Dataset:** IMDb Movie Reviews (Binary Sentiment Analysis)  
**Fine-Tuning Method:** LoRA (Low-Rank Adaptation)  
**Test Set Size:** 500 examples  
**Date:** December 2025

---

## Project Overview

This project evaluates the fine-tuned DistilBERT model from Assignment 7 using comprehensive classification metrics. The evaluation focuses on understanding model performance through quantitative metrics and qualitative error analysis.

---

## Evaluation Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | {accuracy:.4f} ({accuracy*100:.2f}%) |
| **Precision (Macro)** | {precision:.4f} |
| **Recall (Macro)** | {recall:.4f} |
| **F1-Score (Macro)** | {f1:.4f} |

---

## Confusion Matrix Analysis

### Performance by Class

| Class | Correct Predictions | Misclassifications | Accuracy |
|-------|-------------------|-------------------|----------|
| **Negative** | {cm[0,0]}/{cm[0,0] + cm[0,1]} | {cm[0,1]} | {cm_normalized[0,0]:.2f}% |
| **Positive** | {cm[1,1]}/{cm[1,0] + cm[1,1]} | {cm[1,0]} | {cm_normalized[1,1]:.2f}% |

### Key Insights
- The model performs slightly better on **Positive** reviews ({cm_normalized[1,1]:.2f}% accuracy)
- **Negative** reviews are the worst-performing class ({cm_normalized[0,0]:.2f}% accuracy)
- {cm[0,1]} negative reviews were incorrectly classified as positive (false positives)
- {cm[1,0]} positive reviews were incorrectly classified as negative (false negatives)

![Confusion Matrix](confusion_matrix.png)

---

## Why F1-Score (Macro) is the Better Metric

While our dataset is **relatively balanced** ({cm[0,0] + cm[0,1]} negative vs {cm[1,0] + cm[1,1]} positive examples), **F1-Score (Macro)** is still superior to simple accuracy for the following reasons:

### 1. **Balanced View of Precision and Recall**
Accuracy alone doesn't tell us if the model is good at both identifying true positives AND avoiding false positives. F1-Score considers both:
- **Precision**: Of all positive predictions, how many were correct?
- **Recall**: Of all actual positives, how many did we catch?

### 2. **Equal Treatment of Both Classes**
Macro-averaging ensures both classes (Negative and Positive) contribute equally to the final score, preventing bias toward the majority class even in near-balanced datasets.

### 3. **Reveals Hidden Weaknesses**
Our accuracy is {accuracy*100:.2f}%, but F1-Score reveals that:
- Negative class performance: {cm_normalized[0,0]:.2f}% (worse)
- Positive class performance: {cm_normalized[1,1]:.2f}% (better)

If we only looked at accuracy, we would miss that the model struggles more with negative sentiment detection.

### 4. **Production Relevance**
In real-world applications (e.g., review monitoring systems), **both types of errors matter**:
- False Positives: Missing genuinely negative feedback could ignore customer complaints
- False Negatives: Flagging positive reviews as negative wastes review time

F1-Score ensures we optimize for both error types rather than just overall correctness.

---

## Error Analysis: Worst-Performing Class

### Worst-Performing Class: **NEGATIVE**
- **Accuracy:** {cm_normalized[0,0]:.2f}%
- **Misclassifications:** {cm[0,1]} out of {cm[0,0] + cm[0,1]} examples ({cm_normalized[0,1]:.2f}%)

---

### Misclassified Example 1: Boat/Shipwreck Film (Index 17)

**True Label:** Negative  
**Predicted Label:** Positive

**Review Snippet:**
```
"A holiday on a boat, a married couple, an angry waiter and a shipwreck...
I like boobs. No question about that. But when the main character allies 
with whoever happens to have the most fish at the moment, mostly by having 
sex with them and playing the role of the constant victim, my anger just 
rises to a whole new level..."
```

**Why the Model Failed:**
1. **Mixed Tone:** The review starts with casual, conversational language ("I like boobs")
2. **Informal Style:** The humorous/sarcastic opening may signal positive sentiment to the model
3. **Delayed Negativity:** The clear negative sentiment ("my anger just rises") appears AFTER casual language
4. **Ambiguous Cues:** The informal tone might have been weighted more heavily than the explicit anger

**Root Cause:** The model struggles with reviews that mix **casual/humorous language** with **delayed negative sentiment**. The informal conversational style creates ambiguity.

---

### Misclassified Example 2: Kiarostami Art Film (Index 21)

**True Label:** Negative  
**Predicted Label:** Positive

**Review Snippet:**
```
"Coming from Kiarostami, this art-house visual and sound exposition is 
a surprise. For a director known for his narratives and keen observation 
of humans, especially children, this excursion into minimalist 
cinematography begs for questions: Why did he do it? Was it to keep him 
busy during a vacation at the shore?"
```

**Why the Model Failed:**
1. **Subtle Criticism:** Uses rhetorical questions instead of explicit negative statements
2. **Academic Tone:** Professional, analytical language without clear sentiment markers
3. **Ambiguous Phrasing:** "begs for questions" is indirect criticism
4. **Lack of Obvious Negative Words:** No words like "bad," "terrible," "awful"

**Root Cause:** The model struggles with **intellectually-written, analytically-toned reviews** that express criticism through **rhetorical questions** and **subtle disapproval** rather than direct negative language.

---

## Common Pattern in Misclassifications

Both misclassified examples share a critical weakness:

**The model was trained primarily on reviews with EXPLICIT sentiment markers** (e.g., "amazing," "terrible," "loved it," "hated it"). When faced with:
- Subtle, indirect criticism
- Mixed tones (humor + negativity)
- Sophisticated, analytical language
- Rhetorical questions instead of statements

...the model defaults to predicting based on **surface-level tone** rather than **underlying sentiment**.

---

## Recommendations for Improvement

1. **Data Augmentation:** Include more examples with subtle, indirect sentiment
2. **Contextual Training:** Train on reviews where sentiment appears later in the text
3. **Rhetorical Question Handling:** Fine-tune on examples with questions expressing criticism
4. **Tone vs. Sentiment:** Distinguish between conversational tone and actual sentiment

---

## Repository Structure
```
AI-ML-Assignment-8-LLM-Evaluation/
├── llm_evaluation_metrics.ipynb    # Complete evaluation notebook
├── confusion_matrix.png            # Normalized confusion matrix
├── README.md                       # This file
├── requirements.txt                # Package dependencies
└── results/
    └── checkpoint-750/             # Fine-tuned model checkpoint
```

---

## How to Run
```bash
# Activate environment
conda activate llm_finetuning

# Install dependencies (if needed)
pip install -r requirements.txt

# Open Jupyter Notebook
jupyter notebook llm_evaluation_metrics.ipynb
```

---

## Video Demonstration

[YouTube Link - To be added after recording]

---

## Key Learnings

1. **F1-Score reveals class-level performance** that accuracy masks
2. **Macro-averaging** ensures fair evaluation across classes
3. **Error analysis** identifies specific model weaknesses (subtle sentiment, mixed tones)
4. **Confusion matrix** provides actionable insights for model improvement
5. **Binary sentiment analysis** is challenging when reviews use sophisticated language

---

## Comparison with Assignment 7

| Aspect | Assignment 7 | Assignment 8 |
|--------|-------------|-------------|
| **Focus** | Model Training | Model Evaluation |
| **Primary Metric** | Accuracy | F1-Score (Macro) |
| **Analysis Depth** | Basic metrics | Comprehensive + Error Analysis |
| **Deliverable** | Trained model | Performance insights |

"""

