# Incremental Preprocessing Agent

A self-learning preprocessing agent that uses progressive sampling to discover and validate data transformation rules for large-scale datasets.

## Table of Contents

1. [Overview](#1-overview)
2. [5-Stage Workflow](#2-5-stage-workflow)
3. [Core Components](#3-core-components)
4. [Rule Discovery](#4-rule-discovery)
5. [HITL Interface](#5-hitl-interface)
6. [CLI Usage](#6-cli-usage)
7. [API Reference](#7-api-reference)
8. [Examples](#8-examples)

---

## 1. Overview

### Problem Statement

When preprocessing large datasets (100K+ records), traditional approaches face challenges:
- **Memory**: Loading all data at once is impractical
- **Time**: Full analysis is slow
- **Verification**: Cannot manually review all records
- **Confidence**: Need statistical assurance that rules are correct

### Solution: Incremental Sampling

The Incremental Preprocessing Agent solves this by:
1. **Progressive sampling**: Start with 0.1%, grow to 2.5% before full application
2. **Rule learning**: Discover patterns automatically from samples
3. **Confidence tracking**: Measure rule stability across samples
4. **Human-in-the-loop**: Ask humans only for business logic decisions
5. **Bulk application**: Apply validated rules to entire dataset

### Key Benefits

| Benefit | Description |
|---------|-------------|
| **Scalability** | Process any dataset size with constant memory |
| **Confidence** | 98%+ rule stability before full application |
| **Transparency** | Human oversight for critical decisions |
| **Reusability** | Generated C# scripts work for future data |
| **Efficiency** | ~2.5% sample sufficient for rule discovery |

---

## 2. 5-Stage Workflow

```
┌──────────────────────────────────────────────────────────────────────┐
│                         INCREMENTAL WORKFLOW                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Stage 1          Stage 2          Stage 3          Stage 4          │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐     │
│  │ Initial │      │ Pattern │      │  HITL   │      │Confidence│     │
│  │ Explore │ ──▶  │ Expand  │ ──▶  │Decision │ ──▶  │Checkpoint│     │
│  │  0.1%   │      │  0.5%   │      │  1.5%   │      │   2.5%   │     │
│  └─────────┘      └─────────┘      └─────────┘      └─────────┘     │
│       │                │                │                │           │
│       ▼                ▼                ▼                ▼           │
│  Schema           Auto-fix         User            Final            │
│  Discovery        Rules            Decisions       Approval         │
│                                                         │           │
│                                                         ▼           │
│                                                   ┌─────────┐       │
│                                                   │  Bulk   │       │
│                                          Stage 5  │ Process │       │
│                                                   │  100%   │       │
│                                                   └─────────┘       │
│                                                         │           │
│                                                         ▼           │
│                                                   Deliverables      │
└──────────────────────────────────────────────────────────────────────┘
```

### Stage Details

#### Stage 1: Initial Exploration (0.1% sample)

**Purpose**: Discover schema and obvious patterns

**Agent Reasoning**:
> "I'm analyzing 100 records (0.1%) to understand the data structure.
> Found: 15 columns, mixed types in 'error_code', two date formats in 'timestamp'."

**Actions**:
- Schema inference (column names, types)
- Obvious pattern detection (nulls, mixed types)
- Initial rule candidates generation

**Output**: `InitialExplorationReport`

#### Stage 2: Pattern Expansion (0.5% sample)

**Purpose**: Validate patterns, apply auto-fixable rules

**Agent Reasoning**:
> "Analyzing 500 records (cumulative 0.6%). The date format inconsistency is confirmed.
> I can auto-fix this: standardize all dates to ISO-8601 format."

**Actions**:
- Pattern validation with larger sample
- Auto-apply rules that don't need human input
- Identify ambiguous cases needing HITL

**Output**: `PatternExpansionReport`, auto-fix rules applied

#### Stage 3: HITL Decision (1.5% sample)

**Purpose**: Get human decisions for business logic

**Agent Reasoning**:
> "I found 'machine_id' = 'Unknown' in 5% of records. This is a business decision:
> Should I (A) delete these records, or (B) replace with 'Default_M0'?"

**Interaction**:
```
🤖 Agent: "I found 75 records with 'machine_id' = 'Unknown' (5% of sample).
           What should I do?

           (A) Delete these records
           (B) Replace with 'Default_M0'
           (C) Keep as-is
           (D) Other (specify)"

👤 User: "B - Those are from equipment transition period, use Default_M0"

🤖 Agent: "Understood. Rule added: machine_id='Unknown' → 'Default_M0'"
```

**Output**: `HITLDecisionLog`, user-approved rules

#### Stage 4: Confidence Checkpoint (2.5% sample)

**Purpose**: Validate rule stability, get final approval

**Agent Reasoning**:
> "Analyzed 2,500 records (2.5%). Last 1,000 records had no new exceptions.
> Rule stability: 98.5%. Ready for full application."

**Report to User**:
```
📊 Preprocessing Sampling Analysis Report

Current Progress: 2.5% (2,500 / 100,000 records)

Confirmed Rules (12 total):
1. Date format → ISO-8601
2. Unit conversion: lb → kg
3. Missing 'temperature' → median (23.5)
4. machine_id='Unknown' → 'Default_M0'
...

Rule Stability: 98.5% (no new patterns in last 1,000 samples)

Options:
1. [Recommended] Apply to remaining 97,500 records
2. Sample additional 2.5% for more validation
3. Review/modify rule list
```

**Output**: `ConfidenceReport`, final rule set

#### Stage 5: Bulk Processing (100%)

**Purpose**: Apply validated rules to entire dataset

**Agent Reasoning**:
> "Applying 12 confirmed rules to all 100,000 records.
> Monitoring for exceptions. Will stop if error rate exceeds 1%."

**Actions**:
- Apply rules via FilePrepper streaming
- Collect exceptions (outliers that don't fit rules)
- Generate deliverables

**Output**: `Cleaned_Dataset.csv`, `Preprocessing_Logic.cs`, `Exception_Log.json`

---

## 3. Core Components

### 3.1 SamplingStrategy

Controls sample size progression and selection method.

```csharp
public class SamplingStrategy
{
    // Stage-based sample sizes (percentage of total)
    public static readonly double[] StageSampleRates =
        { 0.001, 0.005, 0.015, 0.025, 1.0 };

    // Minimum absolute sample sizes
    public static readonly int[] MinSampleSizes =
        { 100, 500, 1500, 2500, int.MaxValue };

    public SamplingMethod Method { get; set; } = SamplingMethod.Stratified;

    public IEnumerable<T> GetSample<T>(IEnumerable<T> data, int stage);
    public int GetSampleSize(int totalRows, int stage);
}

public enum SamplingMethod
{
    Random,           // Pure random sampling
    Stratified,       // Proportional representation
    Systematic,       // Every Nth record
    ClusterBased      // Group-based sampling
}
```

### 3.2 RuleDiscoveryEngine

Discovers preprocessing rules from data patterns.

```csharp
public class RuleDiscoveryEngine
{
    public IEnumerable<PreprocessingRule> DiscoverRules(
        DataSample sample,
        RuleDiscoveryOptions options);

    public RuleValidationResult ValidateRule(
        PreprocessingRule rule,
        DataSample newSample);
}

public class PreprocessingRule
{
    public string Id { get; set; }
    public string Name { get; set; }
    public RuleType Type { get; set; }
    public string Column { get; set; }
    public string Pattern { get; set; }        // What to match
    public string Transformation { get; set; }  // What to do
    public bool RequiresHITL { get; set; }
    public double Confidence { get; set; }
    public int MatchCount { get; set; }
}
```

### 3.3 ConfidenceCalculator

Measures rule stability and convergence.

```csharp
public class ConfidenceCalculator
{
    public double CalculateRuleConfidence(
        PreprocessingRule rule,
        List<DataSample> samples);

    public bool HasConverged(
        List<PreprocessingRule> rules,
        int samplesWithoutNewRules);

    public ConvergenceReport GetConvergenceReport(
        List<PreprocessingRule> rules,
        List<DataSample> sampleHistory);
}

public class ConvergenceReport
{
    public double OverallConfidence { get; set; }
    public int SamplesSinceLastNewRule { get; set; }
    public bool IsStable { get; set; }
    public List<RuleConfidenceDetail> RuleDetails { get; set; }
}
```

### 3.4 PreprocessingWorkflowState

Manages the 5-stage state machine.

```csharp
public enum WorkflowStage
{
    NotStarted,
    InitialExploration,    // Stage 1
    PatternExpansion,      // Stage 2
    HITLDecision,          // Stage 3
    ConfidenceCheckpoint,  // Stage 4
    BulkProcessing,        // Stage 5
    Completed,
    Failed
}

public class PreprocessingWorkflowState
{
    public WorkflowStage CurrentStage { get; set; }
    public int TotalRecords { get; set; }
    public int ProcessedRecords { get; set; }
    public List<PreprocessingRule> DiscoveredRules { get; set; }
    public List<HITLDecision> UserDecisions { get; set; }
    public ConvergenceReport ConvergenceStatus { get; set; }
    public DateTime StartedAt { get; set; }
    public TimeSpan ElapsedTime { get; set; }
}
```

---

## 4. Rule Discovery

### 4.1 Pattern Detection System

The Rule Discovery Engine uses 7 specialized pattern detectors to identify data quality issues:

| Detector | Pattern Type | Applicable To | Auto-Fixable |
|----------|-------------|---------------|--------------|
| **MissingValueDetector** | MissingValue | All columns | No (requires HITL) |
| **TypeInconsistencyDetector** | TypeInconsistency | String columns | No (requires HITL) |
| **FormatVariationDetector** | FormatVariation | String columns | Yes (dates, numbers) |
| **OutlierDetector** | OutlierAnomaly | Numeric columns | No (requires HITL) |
| **CategoryVariationDetector** | CategoryVariation | String columns | No (requires HITL) |
| **EncodingIssueDetector** | EncodingIssue | String columns | Yes (UTF-8 conversion) |
| **WhitespaceDetector** | WhitespaceIssue | String columns | Yes (trim & collapse) |

**Implementation Details**:
- **Statistical Analysis**: Z-score (threshold: 3.0) and IQR methods for outlier detection
- **String Similarity**: Levenshtein distance for category typo detection
- **Pattern Matching**: GeneratedRegex for encoding issues and format variations
- **Null Detection**: Handles NULL, "N/A", "None", "-", "?" as missing values

### 4.2 Auto-Resolvable Patterns

These patterns are fixed automatically without HITL:

| Rule Type | Detection | Auto-Fix |
|-----------|-----------|----------|
| **DateFormatStandardization** | Multiple date formats detected | Convert to ISO-8601 |
| **EncodingNormalization** | UTF-8/Latin-1 conflicts, mojibake | Normalize to UTF-8 |
| **WhitespaceNormalization** | Leading/trailing spaces, multiple spaces, tabs | Trim and collapse |
| **NumericFormatStandardization** | Number format variations (1,000 vs 1000) | Remove separators, fix decimals |

### 4.3 HITL-Required Patterns

These patterns require human decision:

| Rule Type | Pattern Example | Question to User |
|-----------|----------------|------------------|
| **MissingValueStrategy** | 5% nulls in 'income' | "Delete, impute mean/median, or use default?" |
| **OutlierHandling** | Age = 150 (Z-score > 3.0) | "Keep, remove, or cap at threshold?" |
| **CategoryMapping** | 'status' variations: 'N/A', 'NA', 'Unknown' | "Merge categories or keep separate?" |
| **TypeConversion** | Mixed types in column | "Convert to which type?" |
| **BusinessLogicDecision** | Domain-specific cases | Custom business logic required |

### 4.4 Rule Confidence Calculation

**Formula**:
```
Overall Confidence = Consistency × 0.5 + Coverage × 0.3 + Stability × 0.2

Where:
- Consistency (50%): Rule applies consistently to affected data
- Coverage (30%): Percentage of data covered by rule
- Stability (20%): Rule definition unchanged across samples
```

**Implementation**: `ConfidenceCalculator.cs`
- Cross-sample validation between Stage N-1 and Stage N
- Pattern matching consistency measurement
- Coverage calculation per column
- Stability tracking based on rule signature changes

**Confidence Levels**:
- **High Confidence**: ≥ 98% - Ready for automatic application
- **Medium Confidence**: ≥ 90% - Apply with caution
- **Low Confidence**: < 90% - Requires review or more sampling

### 4.5 Convergence Detection

**Formula**:
```
Change Rate = (NewRules + ModifiedRules + RemovedRules) / BaselineRuleCount
```

**Implementation**: `ConvergenceDetector.cs`
- Default threshold: 2% (0.02) change rate
- Tracks rule additions, modifications, and removals
- Provides detailed convergence metrics and status

**Convergence Criteria**:
- Change rate ≤ threshold across two consecutive samples
- No new patterns discovered in last sample
- Existing rule confidence scores stable (≥ 98%)

When convergence is detected, the system proceeds from sampling to bulk processing (Stage 5).

---

## 5. HITL Interface

### 5.1 Question Types

```csharp
public enum HITLQuestionType
{
    MultipleChoice,      // A, B, C, D options
    YesNo,               // Simple binary
    NumericInput,        // Threshold, percentage
    TextInput,           // Custom value
    Confirmation         // Approve/Reject
}

public class HITLQuestion
{
    public string Id { get; set; }
    public HITLQuestionType Type { get; set; }
    public string Context { get; set; }      // What the agent found
    public string Question { get; set; }     // What to ask
    public List<HITLOption> Options { get; set; }
    public string RecommendedOption { get; set; }
    public string RecommendationReason { get; set; }
}
```

### 5.2 Interactive Prompt Format

```
┌──────────────────────────────────────────────────────────────────┐
│ 🤖 Preprocessing Agent - Decision Required                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│ CONTEXT:                                                          │
│ Analyzed 1,500 records (1.5% of 100,000).                        │
│ Found: 'machine_id' = 'Unknown' in 75 records (5% of sample).    │
│                                                                   │
│ QUESTION:                                                         │
│ How should I handle records with unknown machine_id?             │
│                                                                   │
│ OPTIONS:                                                          │
│   (A) Delete these records                                        │
│   (B) Replace with 'Default_M0' [Recommended]                    │
│   (C) Keep as-is                                                  │
│   (D) Other (specify)                                             │
│                                                                   │
│ RECOMMENDATION: Option B                                          │
│ Reason: Deletion loses 5% of data. 'Default_M0' is a common     │
│         placeholder that preserves data while marking ambiguity. │
│                                                                   │
├──────────────────────────────────────────────────────────────────┤
│ Enter your choice (A/B/C/D):                                      │
└──────────────────────────────────────────────────────────────────┘
```

### 5.3 Final Approval Prompt

```
┌──────────────────────────────────────────────────────────────────┐
│ 📊 Preprocessing Sampling Analysis Report                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│ PROGRESS: 2.5% complete (2,500 / 100,000 records)                │
│ RULE STABILITY: 98.5% (no new patterns in last 1,000 samples)    │
│                                                                   │
│ CONFIRMED RULES (12 total):                                       │
│  1. ✅ timestamp: Multiple formats → ISO-8601                     │
│  2. ✅ weight: lb → kg conversion                                 │
│  3. ✅ temperature: Missing → median (23.5)                       │
│  4. ✅ machine_id: 'Unknown' → 'Default_M0' [User Decision]      │
│  5. ✅ error_code: 'N/A' → null                                   │
│  ... (7 more rules)                                               │
│                                                                   │
│ EXCEPTION ESTIMATE: ~0.1% (100 records may not fit rules)        │
│                                                                   │
├──────────────────────────────────────────────────────────────────┤
│ OPTIONS:                                                          │
│  1. [Recommended] Apply to remaining 97,500 records               │
│  2. Sample additional 2.5% for more validation                    │
│  3. Review/modify rule list                                       │
│  4. Cancel preprocessing                                          │
│                                                                   │
│ Note: Processing will pause if error rate exceeds 1%.            │
├──────────────────────────────────────────────────────────────────┤
│ Enter your choice (1/2/3/4):                                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 6. CLI Usage

### Basic Usage

```bash
# Start incremental preprocessing
mloop agent "preprocess datasets/manufacturing_logs.csv incrementally" --agent preprocessing-expert

# Or use the dedicated command
mloop preprocess datasets/manufacturing_logs.csv --incremental

# With options
mloop preprocess data.csv --incremental \
    --confidence-threshold 0.95 \
    --max-error-rate 0.01 \
    --output-dir ./cleaned
```

### Interactive Session

```bash
$ mloop agent --agent preprocessing-expert --interactive

🤖 Preprocessing Expert Agent
Type 'help' for commands, 'quit' to exit.

> analyze datasets/logs.csv --incremental

📊 Starting Incremental Analysis
Total Records: 100,000
Starting Stage 1: Initial Exploration (100 samples)...

[Stage 1 Complete]
Found: 15 columns, 3 patterns requiring attention

Proceeding to Stage 2: Pattern Expansion...
[500 additional samples]

[Stage 2 Complete]
Auto-fixed: Date format standardization
Auto-fixed: Whitespace normalization
Pending HITL: 2 decisions needed

Proceeding to Stage 3: HITL Decision...

🤖 Decision Required:
[... interactive prompts ...]

> apply

✅ All rules approved. Proceeding to bulk processing...
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--incremental` | false | Enable incremental preprocessing |
| `--confidence-threshold` | 0.98 | Minimum confidence for auto-apply |
| `--max-error-rate` | 0.01 | Stop if error rate exceeds this |
| `--sampling-method` | stratified | random, stratified, systematic |
| `--output-dir` | ./cleaned | Output directory for deliverables |
| `--skip-hitl` | false | Use recommended defaults (no prompts) |
| `--resume` | null | Resume from saved state file |

---

## 7. API Reference

### IncrementalPreprocessingAgent

```csharp
public class IncrementalPreprocessingAgent : ConversationalAgent
{
    // Start incremental preprocessing workflow
    public async Task<PreprocessingWorkflowResult> ProcessIncrementallyAsync(
        string inputPath,
        IncrementalPreprocessingOptions options,
        CancellationToken cancellationToken = default);

    // Process with HITL callback
    public async Task<PreprocessingWorkflowResult> ProcessWithHITLAsync(
        string inputPath,
        IncrementalPreprocessingOptions options,
        Func<HITLQuestion, Task<HITLAnswer>> hitlCallback,
        CancellationToken cancellationToken = default);

    // Get current workflow state
    public PreprocessingWorkflowState GetWorkflowState();

    // Resume from saved state
    public async Task<PreprocessingWorkflowResult> ResumeAsync(
        string stateFilePath,
        CancellationToken cancellationToken = default);
}

public class IncrementalPreprocessingOptions
{
    public double ConfidenceThreshold { get; set; } = 0.98;
    public double MaxErrorRate { get; set; } = 0.01;
    public SamplingMethod SamplingMethod { get; set; } = SamplingMethod.Stratified;
    public string OutputDirectory { get; set; } = "./cleaned";
    public bool SkipHITL { get; set; } = false;
    public int MaxHITLQuestions { get; set; } = 10;
}
```

### PreprocessingWorkflowResult

```csharp
public class PreprocessingWorkflowResult
{
    public bool Success { get; set; }
    public WorkflowStage FinalStage { get; set; }
    public int TotalRecords { get; set; }
    public int ProcessedRecords { get; set; }
    public int ExceptionRecords { get; set; }
    public double ErrorRate { get; set; }
    public List<PreprocessingRule> AppliedRules { get; set; }
    public TimeSpan Duration { get; set; }

    // Output files
    public string CleanedDataPath { get; set; }
    public string PreprocessingScriptPath { get; set; }
    public string ExceptionLogPath { get; set; }
    public string ReportPath { get; set; }
}
```

---

## 8. Examples

### Example 1: Manufacturing Log Preprocessing

**Input**: `manufacturing_logs.csv` (100,000 records)

```csv
timestamp,machine_id,temperature,pressure,error_code,description
2024-01-15,M001,23.5,1.2,100,Normal operation
15/01/2024,Unknown,N/A,1.3,N/A,Pressure overshoot
2024-01-15,M002,24.0,1.1,101,Minor warning
...
```

**Agent Workflow**:

```
Stage 1: Found date format mix, 'Unknown' machine_id, 'N/A' values
Stage 2: Auto-fixed date format to ISO-8601
Stage 3: Asked user about 'Unknown' machine_id → User chose 'Default_M0'
Stage 4: Confidence 98.5%, user approved
Stage 5: Processed 100,000 records, 95 exceptions logged

Deliverables:
- cleaned_manufacturing_logs.csv (99,905 records)
- 01_preprocess_manufacturing.cs (reusable script)
- exceptions.json (95 problematic records)
- preprocessing_report.md
```

### Example 2: Financial Transaction Data

**Input**: `transactions.csv` (500,000 records)

**Key Decisions**:
1. Currency: Mixed USD/EUR → User chose: Convert all to USD
2. Missing amounts: User chose: Flag for review, don't delete
3. Duplicate transactions: User chose: Keep first occurrence

**Result**: 498,234 clean records, 1,766 flagged for review

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-12-30 | Initial design document |

## Related Documents

- [ADR-001: Incremental Preprocessing Architecture](./adr/ADR-001-incremental-preprocessing.md)
- [AI Agent Architecture](./AI-AGENT-ARCHITECTURE.md)
- [AI Agent Usage Guide](./AI-AGENT-USAGE.md)
