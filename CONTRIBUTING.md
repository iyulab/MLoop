# Contributing to MLoop

Thank you for your interest in contributing to MLoop! This document outlines our philosophy, development practices, and how to align your contributions with the project's mission.

---

## Core Philosophy

**Mission**: Excellent MLOps with Minimum Cost

Every contribution should support one or more of these goals:

1. **Reduce Development Cost**: Make ML workflows faster and simpler
2. **Reduce Knowledge Cost**: Enable non-experts to achieve expert-level results
3. **Reduce Operational Cost**: Eliminate infrastructure complexity
4. **Maximize Value**: Maintain production-quality output

---

## Design Principles

### 1. Convention Over Configuration
**Good**: Feature works immediately with intelligent defaults
```csharp
// ✅ Good: Automatic discovery and defaults
mloop train datasets/train.csv price --time 60
```

**Bad**: Requires extensive configuration before use
```csharp
// ❌ Bad: Complex configuration required
mloop train --config complex.yaml --schema schema.json --mappings map.yaml
```

### 2. Simplicity First, Extensibility Optional
**Good**: Simple case is trivial, complex case is possible
```csharp
// ✅ Good: Simple by default, extensible when needed
// Simple: Just works
mloop train data.csv price

// Advanced: Optional customization via scripts
.mloop/scripts/hooks/pre-train.cs
```

**Bad**: Requires understanding complexity for simple cases
```csharp
// ❌ Bad: Must understand pipelines for basic usage
mloop train --pipeline transform-train-evaluate --config pipeline.yaml
```

### 3. Fail Gracefully
**Good**: Extension failures don't break core functionality
```csharp
// ✅ Good: Hook fails, AutoML continues
⚠️  Hook compilation failed: pre-train.cs
Continuing with AutoML training...
✅ Training completed
```

**Bad**: One failure cascades to entire system
```csharp
// ❌ Bad: Hook failure blocks training
❌ Hook compilation failed: pre-train.cs
❌ Training aborted
```

### 4. Filesystem-First
**Good**: All state as files, Git-friendly
```csharp
// ✅ Good: Human-readable files
experiments/exp-001/
├── metadata.json  (Git tracked)
├── metrics.json   (Git tracked)
└── model.zip      (Gitignored)
```

**Bad**: State in databases or binary formats
```csharp
// ❌ Bad: Opaque state storage
.mloop/database.sqlite3
```

---

## Contribution Guidelines

### Before You Start

1. **Check Alignment**: Does your contribution reduce cost (dev/knowledge/ops)?
2. **Check Roadmap**: Is this feature on the [ROADMAP.md](ROADMAP.md)? If not, propose it first.
3. **Check Scope**: Start small, iterate based on feedback.

### Development Process

#### 1. Setup Development Environment
```bash
# Clone repository
git clone https://github.com/yourusername/MLoop.git
cd MLoop

# Install .NET 10.0
dotnet --version  # Should be 10.0+

# Restore packages
dotnet restore

# Build solution
dotnet build

# Run tests
dotnet test
```

#### 2. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

Use branch naming convention:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation only
- `refactor/` - Code refactoring
- `test/` - Test improvements

#### 3. Write Code

**Code Style**:
- Follow existing patterns in codebase
- Use C# 13 features where appropriate
- Enable nullable reference types
- Write XML documentation for public APIs

**Example**:
```csharp
/// <summary>
/// Executes preprocessing scripts in sequential order.
/// </summary>
/// <param name="inputPath">Path to input CSV file</param>
/// <param name="outputDirectory">Directory for intermediate outputs</param>
/// <param name="cancellationToken">Cancellation token</param>
/// <returns>Path to final processed CSV</returns>
public async Task<string> ExecuteAsync(
    string inputPath,
    string outputDirectory,
    CancellationToken cancellationToken = default)
{
    // Implementation
}
```

#### 4. Write Tests

**Test Coverage Requirements**:
- New features: >90% code coverage
- Bug fixes: Add test reproducing the bug
- Refactoring: Maintain or improve existing coverage

**Test Types**:
```csharp
// Unit tests: Fast, isolated
[Fact]
public void GenerateExperimentId_ReturnsValidFormat()
{
    var id = _generator.GenerateId();
    Assert.Matches(@"^exp-\d{3}$", id);
}

// Integration tests: Real filesystem operations
[Fact]
public async Task TrainCommand_WithValidData_CreatesExperiment()
{
    using var tempDir = new TempDirectory();
    var result = await TrainAsync(tempDir.Path, "data.csv");

    Assert.True(Directory.Exists($"{tempDir.Path}/experiments/exp-001"));
}
```

#### 5. Documentation

All new features require:
- [ ] XML documentation comments in code
- [ ] Updates to relevant docs/*.md files
- [ ] Example usage in examples/ or tests
- [ ] CHANGELOG.md entry

**Example Documentation**:
```markdown
## Preprocessing Scripts

MLoop supports optional C# scripts for custom data preprocessing.

### Quick Start

1. Create preprocessing script:
```bash
.mloop/scripts/preprocess/01_join_files.cs
```

2. Implement `IPreprocessingScript`:
```csharp
public class JoinFiles : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext ctx)
    {
        // Your preprocessing logic
    }
}
```

3. Train normally - preprocessing runs automatically:
```bash
mloop train datasets/raw.csv price --time 60
```

### 6. Commit Messages

Follow conventional commits format:
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `refactor`: Code refactoring
- `test`: Test improvements
- `chore`: Build/tooling changes

**Example**:
```
feat(preprocessing): Add IPreprocessingScript extensibility system

- Implement ScriptCompiler with Roslyn integration
- Add PreprocessingEngine for script orchestration
- Support sequential script execution (01_*.cs, 02_*.cs, etc.)
- Auto-compilation with DLL caching for performance

Closes #45
```

#### 7. Submit Pull Request

**PR Checklist**:
- [ ] Code follows project style
- [ ] Tests pass locally (`dotnet test`)
- [ ] New tests written (>90% coverage)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Commit messages follow convention
- [ ] PR description explains motivation and approach

**PR Template**:
```markdown
## Description
Brief description of what this PR does and why.

## Motivation
Why is this change needed? What problem does it solve?

## Philosophy Alignment
How does this reduce cost (development/knowledge/operational)?

## Changes
- Bullet point list of changes
- Include breaking changes (if any)

## Testing
How was this tested?

## Screenshots (if applicable)
Before/after screenshots for UI changes

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

---

## Code Review Guidelines

### For Contributors

**Responding to Feedback**:
- Assume good intent from reviewers
- Ask for clarification if feedback is unclear
- Address all review comments (even if disagreeing)
- Mark conversations as resolved after addressing

### For Reviewers

**Review Philosophy**:
- Align with project mission: Does it reduce cost?
- Maintain simplicity: Could this be simpler?
- Check documentation: Is it clear how to use this?
- Verify tests: >90% coverage, meaningful assertions
- Consider extensibility: Could this be a hook/script instead?

**Review Checklist**:
- [ ] Aligns with "Excellent MLOps with Minimum Cost" mission
- [ ] Maintains Convention Over Configuration principle
- [ ] Fails gracefully (doesn't break existing functionality)
- [ ] Includes tests with >90% coverage
- [ ] Documentation clear and complete
- [ ] Performance impact acceptable (<1ms overhead for optional features)

---

## Common Contribution Patterns

### Adding a New Feature

1. **Propose Feature**: Create GitHub issue with `enhancement` label
   - Describe problem and proposed solution
   - Explain philosophy alignment
   - Get maintainer feedback before coding

2. **Implement Core**: Start with minimal viable implementation
   - Focus on happy path first
   - Keep it simple, avoid premature optimization
   - Ensure it works with intelligent defaults

3. **Add Tests**: Comprehensive test coverage
   - Unit tests for logic
   - Integration tests for workflows
   - E2E tests for CLI commands

4. **Document**: Make it easy to use
   - Code comments for developers
   - User guide for end-users
   - Examples for common scenarios

5. **Iterate**: Respond to feedback
   - Address review comments
   - Refine based on testing
   - Simplify based on user feedback

### Fixing a Bug

1. **Reproduce**: Write failing test first
   - Demonstrates the bug
   - Prevents regression

2. **Fix**: Minimal change to fix issue
   - Don't refactor while fixing
   - Keep changes focused

3. **Verify**: Ensure fix works
   - Failing test now passes
   - No new tests break
   - Edge cases covered

4. **Document**: Update relevant docs
   - CHANGELOG.md entry
   - Fix any incorrect documentation

### Improving Documentation

1. **Identify Gap**: What's confusing or missing?
   - User questions indicate doc gaps
   - Onboarding friction points
   - Complex features without examples

2. **Clarify**: Make it clear and actionable
   - Use concrete examples
   - Show expected output
   - Include common pitfalls

3. **Organize**: Put docs where users look
   - README.md for quick start
   - docs/GUIDE.md for comprehensive usage
   - docs/ARCHITECTURE.md for technical details
   - Code comments for API reference

---

## Special Topics

### Extensibility Features

When adding hooks, metrics, or other extensibility points:

1. **Zero Overhead**: Must have <1ms impact when not used
   ```csharp
   // ✅ Good: Check for existence first
   if (!Directory.Exists(".mloop/scripts")) return; // <1ms

   // ❌ Bad: Always loads scripting engine
   var engine = new ScriptEngine(); // 50ms overhead
   ```

2. **Graceful Degradation**: Script failures don't break core
   ```csharp
   // ✅ Good: Catch and continue
   try {
       await ExecuteHookAsync();
   } catch (Exception ex) {
       _logger.Warning($"Hook failed: {ex.Message}");
       // Continue with AutoML
   }

   // ❌ Bad: Failure propagates
   await ExecuteHookAsync(); // Throws, breaks training
   ```

3. **Type-Safe APIs**: Full C# support with IntelliSense
   ```csharp
   // ✅ Good: Strongly-typed interface
   public interface IPreprocessingScript
   {
       Task<string> ExecuteAsync(PreprocessContext context);
   }

   // ❌ Bad: Stringly-typed API
   Task<object> ExecuteAsync(Dictionary<string, object> context);
   ```

### AI Agent Features

When enhancing AI agents:

1. **Reduce Knowledge Cost**: Make ML accessible to non-experts
   ```csharp
   // ✅ Good: Explains in simple terms
   "Your model achieved 0.85 accuracy, meaning it correctly classifies
    85% of examples. For sentiment analysis, this is good but could be
    improved by collecting more training data."

   // ❌ Bad: Assumes ML expertise
   "F1-score: 0.83, Precision: 0.87, Recall: 0.79"
   ```

2. **Actionable Recommendations**: Specific, not generic
   ```csharp
   // ✅ Good: Specific action
   "Increase --time to 300 seconds to try more algorithms"

   // ❌ Bad: Vague advice
   "Try different hyperparameters"
   ```

3. **Educational**: Teach, don't just execute
   ```csharp
   // ✅ Good: Explains why
   "I'm using 80/20 train-test split because your dataset has 1000 rows,
    which is enough for reliable validation. Smaller datasets might need
    cross-validation instead."

   // ❌ Bad: No explanation
   "Using 80/20 split."
   ```

---

## Getting Help

- **Questions**: GitHub Discussions
- **Bugs**: GitHub Issues with `bug` label
- **Features**: GitHub Issues with `enhancement` label
- **Chat**: Discord server (link in README)

---

## Recognition

Contributors are recognized in:
- CHANGELOG.md for their contributions
- GitHub contributors page
- Release notes for significant features

Thank you for helping make ML accessible to everyone!

---

**Last Updated**: November 13, 2025
**Version**: 1.0
