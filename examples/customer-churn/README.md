# Customer Churn Prediction Example

This example demonstrates using MLoop AI Agents to build a customer churn prediction model.

## Overview

**Goal**: Predict which customers are likely to churn (cancel their subscription)

**Dataset**: Synthetic customer data with features like tenure, monthly charges, contract type

**Task Type**: Binary Classification

## Prerequisites

1. MLoop CLI installed
2. LLM provider configured (see `.env.example`)

## Quick Start

### Step 1: Setup LLM Provider

```bash
# Copy and edit environment file
cp .env.example .env
# Edit .env with your API key
```

### Step 2: Analyze Dataset

```bash
# Use Data Analyst Agent to understand the data
mloop agent chat data-analyst "Analyze datasets/customers.csv. What preprocessing is needed?"
```

Expected output:
- Dataset overview (7,043 rows, 21 columns)
- Feature types and distributions
- Missing value analysis
- ML readiness assessment

### Step 3: Generate Preprocessing Scripts (if needed)

```bash
# Use Preprocessing Expert to handle any data issues
mloop agent chat preprocessing-expert "Handle missing values in TotalCharges column"
```

### Step 4: Get Model Recommendations

```bash
# Use Model Architect for configuration guidance
mloop agent chat model-architect "Binary classification for churn prediction, 7K rows, 27% churn rate"
```

### Step 5: Initialize and Train

```bash
# Use MLOps Manager for end-to-end workflow
mloop agent chat mlops-manager "Initialize project and train model on datasets/customers.csv with target 'Churn'"
```

Or use CLI directly:
```bash
mloop init customer-churn --task binary-classification --label Churn
mloop train datasets/customers.csv --label Churn --time 300 --metric F1Score
```

### Step 6: Evaluate and Deploy

```bash
# Evaluate model performance
mloop agent chat mlops-manager "Evaluate the latest experiment and show metrics"

# Promote to production
mloop agent chat mlops-manager "Promote exp-001 to production"
```

## Dataset Description

| Column | Type | Description |
|--------|------|-------------|
| customerID | string | Unique customer identifier |
| gender | string | Male/Female |
| SeniorCitizen | int | 0/1 |
| Partner | string | Yes/No |
| Dependents | string | Yes/No |
| tenure | int | Months with company |
| PhoneService | string | Yes/No |
| MultipleLines | string | Yes/No/No phone service |
| InternetService | string | DSL/Fiber optic/No |
| OnlineSecurity | string | Yes/No/No internet service |
| OnlineBackup | string | Yes/No/No internet service |
| DeviceProtection | string | Yes/No/No internet service |
| TechSupport | string | Yes/No/No internet service |
| StreamingTV | string | Yes/No/No internet service |
| StreamingMovies | string | Yes/No/No internet service |
| Contract | string | Month-to-month/One year/Two year |
| PaperlessBilling | string | Yes/No |
| PaymentMethod | string | Payment type |
| MonthlyCharges | float | Monthly payment amount |
| TotalCharges | float | Total payment to date |
| Churn | string | Yes/No (TARGET) |

## Expected Results

With 5 minutes training time:
- **F1 Score**: 0.55-0.65
- **Accuracy**: 0.78-0.82
- **AUC**: 0.82-0.87

Key predictors:
1. Contract type (month-to-month highest churn)
2. Tenure (shorter tenure = higher churn)
3. Monthly charges (higher charges = higher churn)
4. Internet service type (fiber optic higher churn)

## File Structure

```
customer-churn/
├── README.md               # This file
├── .env.example            # LLM configuration template
├── mloop.yaml              # MLoop project configuration
├── .gitignore              # Git ignore rules
├── datasets/
│   └── customers.csv       # Training data
└── .mloop/
    └── scripts/
        └── preprocess/     # Preprocessing scripts
```

## Troubleshooting

**Issue**: "TotalCharges has missing values"
```bash
# Solution: Use preprocessing script or let agent handle it
mloop agent chat preprocessing-expert "Fill missing TotalCharges with 0 for new customers"
```

**Issue**: "Class imbalance (27% churn)"
```bash
# Solution: Use F1 or AUC metric instead of accuracy
mloop train datasets/customers.csv --label Churn --metric F1Score
```

**Issue**: "Low recall on churn class"
```bash
# Solution: Ask Model Architect for advice
mloop agent chat model-architect "How to improve recall for minority class?"
```

## Next Steps

1. **Feature Engineering**: Create interaction features
   ```bash
   mloop agent chat preprocessing-expert "Create feature: charges_per_month = TotalCharges / tenure"
   ```

2. **Hyperparameter Tuning**: Increase training time
   ```bash
   mloop train datasets/customers.csv --label Churn --time 600
   ```

3. **Model Comparison**: Try different experiments
   ```bash
   mloop agent chat mlops-manager "Compare all experiments and recommend best model"
   ```

## Related Documentation

- [AI Agent Usage Guide](../../docs/AI-AGENT-USAGE.md)
- [AI Agent Architecture](../../docs/AI-AGENT-ARCHITECTURE.md)
- [MLoop User Guide](../../docs/GUIDE.md)
