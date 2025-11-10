# MLoop Refactoring Guidelines

## Philosophy: v0.x Development Period

MLoop is currently in **v0.x development phase**, which means:

✅ **DO**: Bold structural changes that solve root causes
✅ **DO**: Break backward compatibility if it improves design
✅ **DO**: Refactor aggressively to eliminate technical debt
❌ **DON'T**: Apply quick fixes that leave root causes unaddressed
❌ **DON'T**: Avoid refactoring due to fear of breaking changes

---

## Core Principles

### 1. Root Cause Over Symptoms

**Bad (Symptom Fix)**:
```csharp
// Workaround: Convert DoW column in preprocessing script
// Problem: Every user has to write this script manually
```

**Good (Root Cause Fix)**:
```csharp
// Auto-detect categorical text columns and convert automatically
// Problem: Solved once at framework level
if (IsCategoricalTextColumn(column))
{
    AutoConvertToNumeric(column);
}
```

### 2. User Experience First

**Before Refactoring**:
- Users write manual preprocessing scripts for common patterns
- Schema mismatches in production
- Confusing error messages

**After Refactoring**:
- Framework handles common cases automatically
- Stable schemas guaranteed
- Clear, actionable error messages

### 3. Zero Technical Debt Tolerance

Every discovered issue **must** be resolved at root cause level:
- ❌ Add TODO comments
- ❌ Document workarounds
- ✅ Fix the underlying design
- ✅ Add automated tests

---

## Refactoring Decision Framework

### When to Refactor Aggressively

1. **Data Quality Issues**: Automatic cleaning, validation, type inference
2. **User Pain Points**: Confusing workflows, repetitive tasks
3. **Production Failures**: Schema mismatches, runtime errors
4. **Configuration Bugs**: Silent failures, incorrect behavior

### When to Defer

1. **Performance Optimizations**: Unless blocking (wait for benchmarks)
2. **New Features**: Focus on fixing existing functionality first
3. **Nice-to-Have UX**: Unless causing real user confusion

---

## Current v0.x Refactoring Priorities

### 🔴 P0: Production Blockers

**Issue**: FeaturizeText schema mismatch causes prediction failures
**Root Cause**: ML.NET AutoML can't distinguish categorical vs free-form text
**Solution**: Auto-detect and convert categorical text columns
**Impact**: 100% of users with categorical text affected
**Status**: 🔄 In Progress

**Issue**: ConfigMerger ignores test_split from YAML
**Root Cause**: Config merging logic bug
**Solution**: Fix ConfigMerger to properly merge all values
**Impact**: User confusion, but training still works
**Status**: 🔄 In Progress

### 🟡 P1: User Experience Improvements

**Issue**: Multi-file datasets require manual preprocessing
**Root Cause**: No built-in join support
**Solution**: Common join patterns as built-in transformations
**Impact**: Dataset 007 and similar multi-file scenarios
**Status**: ⏳ Planned

**Issue**: DateTime columns require manual feature extraction
**Root Cause**: No automatic temporal feature engineering
**Solution**: Auto-detect datetime and extract features
**Impact**: All time-series datasets
**Status**: ⏳ Planned

### 🟢 P2: Nice-to-Have

**Issue**: No column type hints in mloop.yaml
**Root Cause**: Design limitation
**Solution**: Add optional column metadata
**Impact**: Advanced users
**Status**: ⏳ Future

---

## Refactoring Workflow

### 1. Identify Root Cause
```
❌ "FeaturizeText causes schema mismatch"
✅ "AutoML uses FeaturizeText for all text, but categorical text needs OneHotEncoding"
```

### 2. Design Solution
```
Options:
A. User writes preprocessing script (symptom fix)
B. Add column type hints in YAML (manual, partial fix)
C. Auto-detect categorical text (automatic, root fix) ✅
```

### 3. Implement Boldly
```csharp
// Before: Let AutoML decide
var pipeline = mlContext.Auto().CreateRegressionExperiment(settings);

// After: Preprocess to guide AutoML
var preprocessed = AutoConvertCategoricalText(data);
var pipeline = mlContext.Auto().CreateRegressionExperiment(settings);
```

### 4. Verify Impact
- Run all dataset tests (004-009)
- Check for schema stability
- Measure performance impact
- Update documentation

### 5. Document Decision
```markdown
## Why We Changed X

**Problem**: Production failures due to Y
**Root Cause**: Design flaw in Z
**Solution**: Refactored to W
**Breaking Changes**: None (or: List breaking changes)
**Migration Guide**: (if needed)
```

---

## Breaking Change Policy (v0.x)

### Allowed Breaking Changes

✅ Configuration file format (mloop.yaml)
✅ Internal APIs (not documented as public)
✅ Command output format
✅ Model file structure
✅ Preprocessing script API (with migration guide)

### Protected Stability

🛡️ Core workflow: `mloop train`, `mloop predict`, `mloop evaluate`
🛡️ Basic mloop.yaml structure (project_name, task, label_column)
🛡️ Prediction output format (CSV with predictions)

---

## Code Quality Standards

### Before Merging Refactoring

1. **All datasets pass**: 004-009 train/predict/evaluate successfully
2. **No new warnings**: Build must be warning-free
3. **Documentation updated**: README, docs/, claudedocs/
4. **Zero technical debt**: No TODO comments, all issues resolved

### Testing Strategy

```bash
# Full regression test suite
cd ML-Resource/004-*/ && mloop train && mloop predict && mloop evaluate
cd ML-Resource/005-*/ && mloop train && mloop predict && mloop evaluate
cd ML-Resource/006-*/ && mloop train && mloop predict && mloop evaluate
cd ML-Resource/007-*/ && mloop train && mloop predict && mloop evaluate
cd ML-Resource/008-*/ && mloop train && mloop predict && mloop evaluate
cd ML-Resource/009-*/ && mloop train && mloop predict && mloop evaluate
```

---

## Example Refactorings

### Example 1: ConfigMerger Fix

**Problem**: test_split shows 0% instead of 20%

**Symptom Fix** ❌:
```csharp
// Just fix the display
var displayValue = config.TestSplit > 0 ? config.TestSplit : 0.2;
```

**Root Cause Fix** ✅:
```csharp
// Fix ConfigMerger to actually merge the value
public MLoopConfig Merge(MLoopConfig yaml, MLoopConfig defaults)
{
    return new MLoopConfig
    {
        Training = new TrainingSettings
        {
            TestSplit = yaml.Training?.TestSplit ?? defaults.Training?.TestSplit ?? 0.2
        }
    };
}
```

### Example 2: Categorical Text Auto-Conversion

**Problem**: FeaturizeText causes schema mismatch

**Symptom Fix** ❌:
```csharp
// User writes preprocessing script every time
// .mloop/scripts/preprocess/01_convert_dow.cs
```

**Root Cause Fix** ✅:
```csharp
// Framework auto-detects and converts
public IDataView PreprocessData(IDataView data)
{
    var categoricalColumns = DetectCategoricalTextColumns(data);

    foreach (var col in categoricalColumns)
    {
        // Convert "Monday" → "1", "Tuesday" → "2", etc.
        data = ConvertCategoricalToNumeric(data, col);
    }

    return data;
}

private bool IsCategoricalTextColumn(IDataView data, string columnName)
{
    var uniqueCount = data.GetColumn<string>(columnName).Distinct().Count();
    var rowCount = data.GetRowCount();

    // Heuristic: If unique values < 10% of rows, likely categorical
    return uniqueCount < rowCount * 0.1 && uniqueCount <= 50;
}
```

---

## Anti-Patterns to Avoid

### ❌ Anti-Pattern 1: TODO-Driven Development
```csharp
// TODO: Fix this properly later
if (column.Contains("DoW"))
{
    // Hardcoded workaround
}
```

### ❌ Anti-Pattern 2: User Documentation as Fix
```markdown
## Known Issues
- If you have DoW column, you need to convert it manually
- Write a preprocessing script like this: ...
```

### ❌ Anti-Pattern 3: Partial Fixes
```csharp
// Only fixes Dataset 009, breaks others
if (projectName == "dataset-009")
{
    ConvertDoW();
}
```

---

## Success Metrics

### Refactoring Success =

1. **Zero manual workarounds**: Users don't need preprocessing scripts for common cases
2. **Zero production failures**: Schema mismatches eliminated
3. **Zero technical debt**: No TODO comments, all root causes addressed
4. **Positive user feedback**: Workflow becomes simpler, not more complex

---

## Version Planning

### v0.1 → v0.2: Data Quality & Stability
- ✅ Automatic comma cleaning
- 🔄 Categorical text auto-conversion
- 🔄 ConfigMerger fixes
- ⏳ DateTime feature extraction

### v0.2 → v0.3: Multi-File Support
- ⏳ Built-in join transformations
- ⏳ Time-series aggregation
- ⏳ Wide-to-long unpivot

### v0.3 → v1.0: Production Readiness
- ⏳ Comprehensive error handling
- ⏳ Performance benchmarks
- ⏳ Full test coverage
- ⏳ Stability guarantees

---

## Conclusion

**Remember**: We're in v0.x. This is the time to:
- Fix root causes, not symptoms
- Refactor boldly, not timidly
- Break things to make them better
- Prioritize user experience over backward compatibility

**The goal**: Ship v1.0 with zero technical debt and rock-solid foundation.
