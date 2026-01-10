# Sentiment Analysis Tutorial

**Difficulty**: Beginner
**Task Type**: Binary Classification (Text)
**Dataset**: Movie reviews (Positive/Negative)
**Time**: 10 minutes

## What You'll Learn

- Train a text classification model
- Understand binary classification
- Work with natural language data
- Interpret precision, recall, and F1 score

---

## The Problem

Given a movie review (text), predict whether the sentiment is:
- **Positive** (good review)
- **Negative** (bad review)

This is a **binary classification** problem - predicting one of two outcomes.

**Real-world applications**:
- Customer feedback analysis
- Product review classification
- Social media monitoring
- Support ticket prioritization

---

## Dataset

**File**: `datasets/reviews.csv` (25 samples)

**Features**:
- `Review` - Movie review text

**Label**: `Sentiment` (Positive/Negative)

**Examples**:
- "This movie was absolutely fantastic!" → Positive
- "Terrible movie. Waste of time." → Negative

---

## Step 1: Initialize Project

```bash
cd examples/tutorials/sentiment-analysis
mloop init
```

---

## Step 2: Train Model

```bash
mloop train
```

**What happens**:
1. ML.NET automatically extracts text features using **FeaturizeText**
2. Converts words into numerical vectors (TF-IDF, n-grams)
3. Tries different binary classification algorithms
4. Selects best model based on accuracy

**Expected output**:
```
Training Configuration
─────────────────────
Model         default
Task          binary-classification
Data File     datasets/reviews.csv
Label Column  Sentiment
Time Limit    60s
Metric        accuracy

Trial 1: SdcaLogisticRegressionBinary - accuracy=0.9200
Trial 2: FastTreeBinary - accuracy=0.8800
Trial 3: LightGbmBinary - accuracy=0.9600
...

Training Complete!
─────────────────
Best Trainer: LightGbmBinary
Training Time: 58.32s

METRIC              VALUE
ACCURACY            0.9600
AUC                 0.9850
F1 SCORE            0.9583
PRECISION           0.9583
RECALL              0.9583

Model promoted to production!
```

---

## Step 3: Understand Binary Classification Metrics

### Accuracy (0.96)
- Overall correctness: 96% of predictions are correct
- Good starting point, but not enough for imbalanced data

### AUC - Area Under Curve (0.9850)
- Measures model's ability to distinguish classes
- Range: 0.5 (random) to 1.0 (perfect)
- 0.985 = excellent discrimin ation

### Precision (0.9583)
- Of all Positive predictions, 95.83% were actually Positive
- "How many predicted Positives were correct?"
- Important when false positives are costly

### Recall (0.9583)
- Of all actual Positive reviews, 95.83% were correctly identified
- "How many actual Positives did we find?"
- Important when missing positives is costly

### F1 Score (0.9583)
- Harmonic mean of Precision and Recall
- Balanced measure of model performance
- Use when you care about both false positives and false negatives

**Example scenario**:
- Spam filter: High precision (few good emails marked as spam)
- Disease screening: High recall (find all sick patients)

---

## Step 4: Make Predictions

Create `predict.csv`:

```csv
Review
This film exceeded all my expectations! Loved it.
Waste of money. Very disappointed with the ending.
Average movie. Nothing special but not terrible either.
```

Predict:

```bash
mloop predict predict.csv
```

**Output**:
```
Predictions
────────────────────────────────────────────────────────────────
Review                                              PredictedLabel  Probability
This film exceeded all my expectations! Loved it.   Positive        0.9823
Waste of money. Very disappointed with the ending.  Negative        0.9156
Average movie. Nothing special but not terrible...  Negative        0.6234

Results saved to: predict_predictions.csv
```

**Understanding Probability**:
- 0.9823: Model is 98.23% confident this is Positive
- 0.6234: Model is only 62.34% confident this is Negative (borderline case)

---

## Step 5: Analyze Confusing Cases

Some reviews are harder to classify:

```csv
Review
Not bad for a low-budget film.
Could have been better but still enjoyable.
I didn't hate it but wouldn't recommend it.
```

These are **ambiguous** because:
- Mixed sentiment ("not bad" = slightly positive)
- Qualified statements ("enjoyable" but "could have been better")
- Neutral tone

**Try predicting**:
```bash
mloop predict ambiguous.csv
```

Look at the `Probability` column:
- High confidence (>0.9): Clear sentiment
- Low confidence (0.5-0.7): Ambiguous cases
- Threshold (0.5): Decision boundary

---

## Step 6: Evaluate Model

```bash
mloop evaluate datasets/reviews.csv
```

**Output**:
```
Evaluation Results
──────────────────
METRIC              VALUE
ACCURACY            0.9600
AUC                 0.9850
F1 SCORE            0.9583
PRECISION           0.9583
RECALL              0.9583

Confusion Matrix:
              Predicted Neg  Predicted Pos
Actual Neg    13             1
Actual Pos    0              11

True Positives (TP):  11 - Correctly identified Positive
False Positives (FP): 1  - Incorrectly labeled as Positive
True Negatives (TN):  13 - Correctly identified Negative
False Negatives (FN): 0  - Missed Positive reviews
```

**Reading the confusion matrix**:
- Perfect Positive recall (0 false negatives)
- 1 false positive (a Negative review predicted as Positive)

---

## Key Concepts Learned

### 1. **Binary Classification**
- Two-class prediction (Yes/No, Positive/Negative, True/False)
- Different from multiclass (3+ categories) and regression (continuous values)

### 2. **Text Featurization**
ML.NET automatically converts text to numbers using:
- **Tokenization**: Split into words
- **N-grams**: Word combinations ("not good", "very bad")
- **TF-IDF**: Term frequency-inverse document frequency
- **Word embeddings**: Semantic representations

### 3. **Evaluation Metrics**
Choose based on your needs:
- **Balanced data**: Use Accuracy
- **Imbalanced data**: Use F1 Score or AUC
- **Cost-sensitive**: Optimize Precision or Recall
- **Ranking**: Use AUC

### 4. **Confidence Scores**
- `Probability` shows model's certainty
- Use threshold adjustment for different needs:
  - Conservative (0.7): Only high-confidence predictions
  - Aggressive (0.3): Catch more positives, accept more errors

---

## Common Issues & Solutions

### Issue: Low Accuracy (<70%)

**Possible causes**:
1. **Insufficient training data** (25 samples is very small)
2. **Ambiguous labeling** (subjective sentiment)
3. **Short time limit**

**Solutions**:
```bash
# Increase training time
mloop train --time 120

# Check data quality
mloop train --analyze-data

# Add more diverse training examples
```

### Issue: Imbalanced Classes

If you have 90% Positive, 10% Negative:

**Check class distribution**:
```bash
# Analyze data quality
mloop train --analyze-data
```

**Optimize for F1 instead of Accuracy**:
```bash
mloop train --metric f1_score
```

### Issue: Model Confused by Negations

Text: "Not good at all" → Incorrectly predicted as Positive

**Causes**:
- "good" is a positive word
- Model doesn't understand "not good" = negative

**Solutions**:
- Add more examples with negations to training data
- Consider using advanced NLP features (requires custom preprocessing)

---

## Advanced: Custom Preprocessing

Add text preprocessing script (`.mloop/scripts/preprocess/01_text_cleaning.cs`):

```csharp
using MLoop.Extensibility.Preprocessing;

public class TextCleaningScript : IPreprocessingScript
{
    public async Task<PreprocessingResult> ExecuteAsync(PreprocessingContext context)
    {
        var lines = await File.ReadAllLinesAsync(context.InputPath);
        var cleaned = lines.Select(line =>
        {
            var parts = line.Split(',');
            if (parts.Length < 2) return line;

            var review = parts[0]
                .ToLower()                    // Lowercase
                .Replace("!", "")             // Remove punctuation
                .Replace("?", "")
                .Replace(".", "")
                .Trim();

            return $"{review},{parts[1]}";
        });

        var outputPath = context.GetTempPath("cleaned.csv");
        await File.WriteAllLinesAsync(outputPath, cleaned);

        return new PreprocessingResult
        {
            OutputPath = outputPath,
            Success = true
        };
    }
}
```

---

## Next Steps

### 1. Improve Model Performance

**Collect more data**:
- At least 100+ examples per class
- Cover diverse review styles
- Include edge cases (sarcasm, mixed sentiment)

**Experiment with metrics**:
```bash
mloop train --metric auc        # For ranking
mloop train --metric f1_score   # For balanced precision/recall
```

### 2. Deploy to Production

```bash
# Serve model as REST API
mloop serve

# Test API
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Review": "Amazing movie! Highly recommend."}'
```

### 3. Monitor Performance

```bash
# Add post-train hook for monitoring
# .mloop/scripts/hooks/02_post_train_monitoring.cs
```

### 4. Try Related Tutorials

- **Regression**: `examples/tutorials/housing-prices/` - Predict continuous values
- **Multiclass**: `examples/tutorials/iris-classification/` - 3+ categories
- **Advanced**: `examples/tutorials/multi-file-workflow/` - Complex preprocessing

---

## Summary

**You just**:
✅ Trained a text classification model (96% accuracy)
✅ Learned binary classification evaluation metrics
✅ Understood precision vs recall trade-offs
✅ Made predictions with confidence scores

**Lines of code written**: 0 (pure configuration!)

**Real-world impact**: This same approach works for:
- Customer support ticket routing
- Email spam detection
- Product review analysis
- Social media sentiment tracking
