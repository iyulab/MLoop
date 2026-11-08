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

**File**: `datasets/reviews.csv` (201 samples: 98 Positive, 103 Negative)

The reviews are short, generated from pools of opening, middle and closing phrases (plus a few hand-written mixed ones), so they are far easier than real reviews: expect near-perfect scores here and much less on your own text. It is the size that matters for the tutorial — a binary model needs a few dozen rows of each class in its held-out share to be judged at all (with 25 rows it holds out five, and the quality gate refuses to promote a model that cannot beat chance).

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
mloop init .
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

**Output** (the end of the run — the search prints a progress line per trial before it):
```
── Training Complete! ──────────────────────────────────────────────────────────

> Model: default
> Experiment ID: exp-001
> Best Trainer: LbfgsLogisticRegression
> Training Time: 60.13s

┌────────────────────┬────────┐
│ Metric             │ Value  │
├────────────────────┼────────┤
│ AUC                │ 1.0000 │
│ AUPRC              │ 1.0000 │
│ RECALL             │ 1.0000 │
│ NEGATIVE PRECISION │ 1.0000 │
│ ACCURACY           │ 0.9756 │
│ F1 SCORE           │ 0.9756 │
│ PRECISION          │ 0.9524 │
│ NEGATIVE RECALL    │ 0.9524 │
│ ECE                │ 0.0499 │
└────────────────────┴────────┘

Model saved to: models/default/staging/exp-001/model.zip
Model promoted to production!
```

This is one real run; the search is time-limited, so your trainer and numbers will differ a little. The held-out share is 41 reviews, so one wrong answer moves accuracy by about 2.4 points.

---

## Step 3: Understand Binary Classification Metrics

### Accuracy (0.9756)
- Overall correctness: 40 of the 41 held-out reviews were right
- Good starting point, but not enough for imbalanced data

### AUC - Area Under Curve (1.0000)
- Measures model's ability to rank Positive reviews above Negative ones
- Range: 0.5 (random) to 1.0 (perfect)
- 1.0 here because the generated reviews are easy to tell apart — real text rarely does this well

### Precision (0.9524)
- Of all Positive predictions, 95.24% were actually Positive (20 of 21)
- "How many predicted Positives were correct?"
- Important when false positives are costly

### Recall (1.0000)
- Of all actual Positive reviews, 100% were correctly identified
- "How many actual Positives did we find?"
- Important when missing positives is costly

### F1 Score (0.9756)
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

**Output** (saved to `predictions/default-predictions-<timestamp>.csv`):
```
Review,PredictedLabel,RawScore,Probability,Confidence
This film exceeded all my expectations! Loved it.,Positive,1.7501705,0.8051615,0.805161
Waste of money. Very disappointed with the ending.,Negative,-1.7184584,0.15416738,0.845833
Average movie. Nothing special but not terrible either.,Negative,-3.3455496,0.04045078,0.959549
```

**Understanding the columns**:
- `Probability` is the chance of the Positive class: 0.805 for the first review, 0.154 for the second
- `Confidence` is the probability of the class it chose: 0.806 Positive, then 0.846 Negative
- The third review (neutral wording, labelled Negative in the training data) gets the most confident Negative

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

This is what this model answered:
```
Review,PredictedLabel,RawScore,Probability,Confidence
Not bad for a low-budget film.,Negative,-1.0608093,0.24777502,0.752225
Could have been better but still enjoyable.,Negative,-1.1378047,0.23508877,0.764911
I didn't hate it but wouldn't recommend it.,Negative,-1.4277289,0.1914405,0.80856
```

All three come out Negative, at 75-81% confidence, and the first two read as mildly positive to a person. The model
only knows the words it was trained on — "not bad" and "enjoyable" never appear in a positive review of the training
file — so a confident answer is not a correct one. Lower confidence (0.5-0.7) flags the doubtful cases; the threshold
that turns a probability into a label is 0.5 unless you set one (`mloop threshold`).

---

## Step 6: Evaluate Model

```bash
mloop evaluate datasets/reviews.csv
```

**Output** (trimmed):
```
── Evaluation Results ──────────────────────────────────────────────────────────

│ Metric    │ Training │ Test   │ Difference │
│ f1_score  │ 0.9756   │ 0.9949 │ +0.0193    │
│ precision │ 0.9524   │ 0.9899 │ +0.0375    │
│ recall    │ 1.0000   │ 1.0000 │ +0.0000    │
│ auc       │ 1.0000   │ 1.0000 │ +0.0000    │

Confusion (rows: actual · columns: predicted)
│          │ Negative │ Positive │
│ Negative │      102 │        1 │
│ Positive │        0 │       98 │
```

The "Test" column here is the whole file, rows the model trained on included, so it reads better than the held-out
"Training" column above it (which is what the run measured on rows it had not seen).

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
1. **Insufficient training data** (a few dozen rows is very small: with 25 the held-out share is five reviews, and a model trained on so few can end up answering one class only)
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
using System.IO;
using System.Linq;
using System.Threading.Tasks;

public class TextCleaningScript : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext ctx)
    {
        var lines = await File.ReadAllLinesAsync(ctx.InputPath);
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

        var outputPath = Path.Combine(ctx.OutputDirectory, "01_cleaned.csv");
        await File.WriteAllLinesAsync(outputPath, cleaned);

        return outputPath;
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
✅ Trained a text classification model (about 98% on a small, generated sample — your own text will score lower)
✅ Learned binary classification evaluation metrics
✅ Understood precision vs recall trade-offs
✅ Made predictions with confidence scores

**Lines of code written**: 0 (pure configuration!)

**Real-world impact**: This same approach works for:
- Customer support ticket routing
- Email spam detection
- Product review analysis
- Social media sentiment tracking
