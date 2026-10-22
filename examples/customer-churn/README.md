# Customer Churn Prediction Example

A binary-classification project: predict whether a customer cancels their subscription.

**Dataset**: `datasets/customers.csv` — 50 synthetic customers (14 churned) with tenure, charges, contract
type and services. It is small on purpose, so a walk-through runs in seconds; expect MLoop to warn about
the size, and do not read much into the metrics.

## Walk-through

```bash
cp -r examples/customer-churn ~/churn && cd ~/churn

# The folder already has its mloop.yaml (task, label, metric); init adds the .mloop/ marker and keeps it.
mloop init .

# Train the default model on data.train from mloop.yaml — 300 s; pass --time 30 for a quick run
mloop train --time 30

# Look at the experiment, then score rows with the production model
mloop list
mloop predict datasets/customers.csv
```

The first trained model is promoted to production. Train again (for example with a longer `--time`) and
MLoop promotes the new experiment only when it beats production on the project's metric (`f1_score` here);
`mloop compare exp-001 exp-002` shows the two side by side.

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

## What to look at

- `customerID` is excluded from the features automatically — every value is distinct, so it is an identifier.
- The label is imbalanced (28% churn), which is why `mloop.yaml` optimizes `f1_score` rather than accuracy.
- `models/default/staging/exp-001/report.md` is the one-page summary of the run.

## Related Documentation

- [MLoop User Guide](../../docs/GUIDE.md)
- [Using MLoop from AI agents](../../docs/AI-AGENTS.md)
