using System.Text.Json;
using Microsoft.ML;

namespace MLoop.Core.Pipeline;

/// <summary>
/// Executes ML pipelines defined in YAML
/// </summary>
public class PipelineExecutor
{
    private readonly MLContext _mlContext;
    private readonly Dictionary<string, object> _context = new();
    private readonly Dictionary<string, IPipelineStepHandler> _handlers = new();

    public PipelineExecutor(MLContext mlContext)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        RegisterDefaultHandlers();
    }

    public PipelineExecutor(MLContext mlContext, IEnumerable<IPipelineStepHandler> handlers)
    {
        _mlContext = mlContext ?? throw new ArgumentNullException(nameof(mlContext));
        foreach (var handler in handlers)
        {
            _handlers[handler.StepType.ToLowerInvariant()] = handler;
        }
    }

    private void RegisterDefaultHandlers()
    {
        // Register placeholder handlers - can be overridden by passing handlers to constructor
        _handlers["preprocess"] = new DefaultPreprocessHandler();
        _handlers["train"] = new DefaultTrainHandler();
        _handlers["evaluate"] = new DefaultEvaluateHandler();
        _handlers["predict"] = new DefaultPredictHandler();
        _handlers["promote"] = new DefaultPromoteHandler();
    }

    public async Task<PipelineResult> ExecuteAsync(
        PipelineDefinition pipeline,
        Action<string>? logger = null,
        CancellationToken cancellationToken = default)
    {
        var startTime = DateTime.UtcNow;
        var stepResults = new List<StepResult>();
        var pipelineStatus = PipelineStatus.Completed;
        string? pipelineError = null;

        logger?.Invoke($"🚀 Starting pipeline: {pipeline.Name}");

        if (!string.IsNullOrEmpty(pipeline.Description))
        {
            logger?.Invoke($"📝 {pipeline.Description}");
        }

        // Initialize variables
        if (pipeline.Variables != null)
        {
            foreach (var (key, value) in pipeline.Variables)
            {
                _context[key] = value;
            }
        }

        try
        {
            // Group steps for parallel execution
            var stepGroups = GroupStepsForParallelExecution(pipeline.Steps);

            int totalSteps = pipeline.Steps.Count;
            int currentStep = 0;

            foreach (var group in stepGroups)
            {
                if (group.Count == 1)
                {
                    // Sequential execution
                    var step = group[0];
                    currentStep++;
                    logger?.Invoke($"\n📦 Step {currentStep}/{totalSteps}: {step.Name} ({step.Type})");

                    var stepResult = await ExecuteStepAsync(step, logger, cancellationToken);
                    stepResults.Add(stepResult);

                    if (stepResult.Status == StepStatus.Failed)
                    {
                        if (step.ContinueOnError)
                        {
                            logger?.Invoke($"⚠️  Step failed but continuing (continue_on_error: true)");
                            pipelineStatus = PipelineStatus.PartiallyCompleted;
                        }
                        else
                        {
                            logger?.Invoke($"❌ Pipeline failed at step: {step.Name}");
                            pipelineStatus = PipelineStatus.Failed;
                            pipelineError = stepResult.Error;
                            break;
                        }
                    }
                    else
                    {
                        logger?.Invoke($"✅ Step completed successfully");
                    }
                }
                else
                {
                    // Parallel execution
                    logger?.Invoke($"\n⚡ Executing {group.Count} steps in parallel...");

                    var tasks = group.Select(async step =>
                    {
                        currentStep++;
                        logger?.Invoke($"📦 Starting parallel step: {step.Name} ({step.Type})");
                        return await ExecuteStepAsync(step, logger, cancellationToken);
                    }).ToList();

                    var results = await Task.WhenAll(tasks);
                    stepResults.AddRange(results);

                    // Check for failures
                    foreach (var (step, result) in group.Zip(results))
                    {
                        if (result.Status == StepStatus.Failed)
                        {
                            if (step.ContinueOnError)
                            {
                                logger?.Invoke($"⚠️  Step {step.Name} failed but continuing");
                                pipelineStatus = PipelineStatus.PartiallyCompleted;
                            }
                            else
                            {
                                logger?.Invoke($"❌ Pipeline failed at parallel step: {step.Name}");
                                pipelineStatus = PipelineStatus.Failed;
                                pipelineError = result.Error;
                                break;
                            }
                        }
                        else
                        {
                            logger?.Invoke($"✅ Parallel step {step.Name} completed");
                        }
                    }

                    if (pipelineStatus == PipelineStatus.Failed)
                        break;
                }
            }
        }
        catch (Exception ex)
        {
            pipelineStatus = PipelineStatus.Failed;
            pipelineError = ex.Message;
            logger?.Invoke($"❌ Pipeline failed: {ex.Message}");
        }

        var endTime = DateTime.UtcNow;
        var duration = endTime - startTime;

        logger?.Invoke($"\n{'='}{string.Join("", Enumerable.Repeat("=", 50))}");
        logger?.Invoke($"✨ Pipeline {(pipelineStatus == PipelineStatus.Completed ? "completed" : pipelineStatus.ToString().ToLowerInvariant())} in {duration.TotalSeconds:F2}s");

        return new PipelineResult
        {
            PipelineName = pipeline.Name,
            StartTime = startTime,
            EndTime = endTime,
            Duration = duration,
            Status = pipelineStatus,
            StepResults = stepResults,
            Error = pipelineError
        };
    }

    private async Task<StepResult> ExecuteStepAsync(
        PipelineStep step,
        Action<string>? logger,
        CancellationToken cancellationToken)
    {
        var startTime = DateTime.UtcNow;
        var status = StepStatus.Completed;
        Dictionary<string, object>? outputs = null;
        string? error = null;

        try
        {
            cancellationToken.ThrowIfCancellationRequested();

            // Resolve variable substitutions in parameters
            var resolvedParams = ResolveVariables(step.Parameters);

            // Get handler for step type
            var stepType = step.Type.ToLowerInvariant();
            if (!_handlers.TryGetValue(stepType, out var handler))
            {
                throw new NotSupportedException($"Unknown step type: {step.Type}");
            }

            // Execute step using handler
            outputs = await handler.ExecuteAsync(resolvedParams, logger, cancellationToken);

            // Store outputs in context for subsequent steps
            if (outputs != null)
            {
                foreach (var (key, value) in outputs)
                {
                    _context[$"{step.Name}.{key}"] = value;
                }
            }
        }
        catch (Exception ex)
        {
            status = StepStatus.Failed;
            error = ex.Message;
            logger?.Invoke($"Error: {ex.Message}");
        }

        var endTime = DateTime.UtcNow;

        return new StepResult
        {
            StepName = step.Name,
            StepType = step.Type,
            StartTime = startTime,
            EndTime = endTime,
            Duration = endTime - startTime,
            Status = status,
            Outputs = outputs,
            Error = error
        };
    }

    private Dictionary<string, object> ResolveVariables(Dictionary<string, object> parameters)
    {
        var resolved = new Dictionary<string, object>();

        foreach (var (key, value) in parameters)
        {
            if (value is string str && str.StartsWith("$"))
            {
                var varName = str[1..];
                resolved[key] = _context.ContainsKey(varName) ? _context[varName] : value;
            }
            else
            {
                resolved[key] = value;
            }
        }

        return resolved;
    }

    private List<List<PipelineStep>> GroupStepsForParallelExecution(List<PipelineStep> steps)
    {
        var groups = new List<List<PipelineStep>>();
        var completedSteps = new HashSet<string>();
        var remainingSteps = new Queue<PipelineStep>(steps);

        while (remainingSteps.Count > 0)
        {
            var currentGroup = new List<PipelineStep>();
            var stepsToRequeue = new List<PipelineStep>();

            while (remainingSteps.Count > 0)
            {
                var step = remainingSteps.Dequeue();

                // Check if dependencies are met
                bool dependenciesMet = step.DependsOn == null ||
                    step.DependsOn.All(dep => completedSteps.Contains(dep));

                if (!dependenciesMet)
                {
                    stepsToRequeue.Add(step);
                    continue;
                }

                // Check if condition is met (if present)
                if (step.Condition != null && !EvaluateCondition(step.Condition))
                {
                    // Skip this step - condition not met
                    completedSteps.Add(step.Name);
                    continue;
                }

                // Check if this step can be parallelized with current group
                if (currentGroup.Count == 0 || (step.Parallel && currentGroup.All(s => s.Parallel)))
                {
                    currentGroup.Add(step);
                }
                else
                {
                    stepsToRequeue.Add(step);
                }
            }

            if (currentGroup.Count > 0)
            {
                groups.Add(currentGroup);
                foreach (var step in currentGroup)
                {
                    completedSteps.Add(step.Name);
                }
            }

            // Re-queue steps that weren't ready
            foreach (var step in stepsToRequeue)
            {
                remainingSteps.Enqueue(step);
            }

            // Prevent infinite loop if no progress
            if (currentGroup.Count == 0 && remainingSteps.Count > 0)
            {
                // Force sequential execution for remaining steps
                var forcedStep = remainingSteps.Dequeue();
                groups.Add(new List<PipelineStep> { forcedStep });
                completedSteps.Add(forcedStep.Name);
            }
        }

        return groups;
    }

    private bool EvaluateCondition(StepCondition condition)
    {
        try
        {
            // Parse expression to extract variable and comparison
            var expression = condition.Expression;

            // Simple expression parser: "$variable operator value"
            // e.g., "$train_step.metric > 0.9"
            var parts = expression.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 3)
                return false;

            var variableName = parts[0].TrimStart('$');
            var operatorStr = parts[1];
            var expectedValue = string.Join(" ", parts.Skip(2));

            // Get actual value from context
            if (!_context.TryGetValue(variableName, out var actualValue))
                return false;

            // Determine operator from string
            var op = operatorStr switch
            {
                "==" or "=" => ConditionOperator.Equals,
                "!=" => ConditionOperator.NotEquals,
                ">" => ConditionOperator.GreaterThan,
                "<" => ConditionOperator.LessThan,
                ">=" => ConditionOperator.GreaterThanOrEqual,
                "<=" => ConditionOperator.LessThanOrEqual,
                "contains" => ConditionOperator.Contains,
                "not_contains" => ConditionOperator.NotContains,
                _ => condition.Operator
            };

            return EvaluateOperator(actualValue, op, expectedValue);
        }
        catch
        {
            return false;
        }
    }

    private bool EvaluateOperator(object actualValue, ConditionOperator op, string expectedValue)
    {
        var actualStr = actualValue?.ToString() ?? "";

        return op switch
        {
            ConditionOperator.Equals => actualStr.Equals(expectedValue, StringComparison.OrdinalIgnoreCase),
            ConditionOperator.NotEquals => !actualStr.Equals(expectedValue, StringComparison.OrdinalIgnoreCase),
            ConditionOperator.GreaterThan => CompareNumeric(actualValue, expectedValue) > 0,
            ConditionOperator.LessThan => CompareNumeric(actualValue, expectedValue) < 0,
            ConditionOperator.GreaterThanOrEqual => CompareNumeric(actualValue, expectedValue) >= 0,
            ConditionOperator.LessThanOrEqual => CompareNumeric(actualValue, expectedValue) <= 0,
            ConditionOperator.Contains => actualStr.Contains(expectedValue, StringComparison.OrdinalIgnoreCase),
            ConditionOperator.NotContains => !actualStr.Contains(expectedValue, StringComparison.OrdinalIgnoreCase),
            _ => false
        };
    }

    private int CompareNumeric(object actualValue, string expectedValue)
    {
        if (actualValue is double d1 && double.TryParse(expectedValue, out var d2))
            return d1.CompareTo(d2);

        if (actualValue is int i1 && int.TryParse(expectedValue, out var i2))
            return i1.CompareTo(i2);

        if (actualValue is float f1 && float.TryParse(expectedValue, out var f2))
            return f1.CompareTo(f2);

        // Try to parse both as doubles
        if (double.TryParse(actualValue?.ToString(), out var dActual) &&
            double.TryParse(expectedValue, out var dExpected))
            return dActual.CompareTo(dExpected);

        return 0;
    }

}

// Default placeholder handlers - can be replaced with real implementations via DI

internal class DefaultPreprocessHandler : IPipelineStepHandler
{
    public string StepType => "preprocess";

    public Task<Dictionary<string, object>> ExecuteAsync(
        Dictionary<string, object> parameters,
        Action<string>? logger,
        CancellationToken cancellationToken)
    {
        logger?.Invoke($"   Input: {parameters.GetValueOrDefault("input_file")}");
        logger?.Invoke($"   Output: {parameters.GetValueOrDefault("output_file")}");

        // Placeholder: In real implementation, execute FilePrepper scripts
        return Task.FromResult(new Dictionary<string, object>
        {
            ["output_file"] = parameters.GetValueOrDefault("output_file", "processed_data.csv") ?? "processed_data.csv"
        });
    }
}

internal class DefaultTrainHandler : IPipelineStepHandler
{
    public string StepType => "train";

    public Task<Dictionary<string, object>> ExecuteAsync(
        Dictionary<string, object> parameters,
        Action<string>? logger,
        CancellationToken cancellationToken)
    {
        logger?.Invoke($"   Data: {parameters.GetValueOrDefault("data_file")}");
        logger?.Invoke($"   Label: {parameters.GetValueOrDefault("label_column")}");
        logger?.Invoke($"   Time: {parameters.GetValueOrDefault("training_time", 60)}s");

        // Placeholder: In real implementation, call AutoML training
        return Task.FromResult(new Dictionary<string, object>
        {
            ["experiment_id"] = "exp-pipeline-001",
            ["metric"] = 0.95
        });
    }
}

internal class DefaultEvaluateHandler : IPipelineStepHandler
{
    public string StepType => "evaluate";

    public Task<Dictionary<string, object>> ExecuteAsync(
        Dictionary<string, object> parameters,
        Action<string>? logger,
        CancellationToken cancellationToken)
    {
        logger?.Invoke($"   Model: {parameters.GetValueOrDefault("model")}");
        logger?.Invoke($"   Test data: {parameters.GetValueOrDefault("test_file")}");

        // Placeholder: In real implementation, evaluate model
        return Task.FromResult(new Dictionary<string, object>
        {
            ["r_squared"] = 0.95,
            ["rmse"] = 12.5
        });
    }
}

internal class DefaultPredictHandler : IPipelineStepHandler
{
    public string StepType => "predict";

    public Task<Dictionary<string, object>> ExecuteAsync(
        Dictionary<string, object> parameters,
        Action<string>? logger,
        CancellationToken cancellationToken)
    {
        logger?.Invoke($"   Model: {parameters.GetValueOrDefault("model", "production")}");
        logger?.Invoke($"   Input: {parameters.GetValueOrDefault("input_file")}");

        // Placeholder: In real implementation, run predictions
        return Task.FromResult(new Dictionary<string, object>
        {
            ["prediction_file"] = "predictions.csv"
        });
    }
}

internal class DefaultPromoteHandler : IPipelineStepHandler
{
    public string StepType => "promote";

    public Task<Dictionary<string, object>> ExecuteAsync(
        Dictionary<string, object> parameters,
        Action<string>? logger,
        CancellationToken cancellationToken)
    {
        logger?.Invoke($"   Experiment: {parameters.GetValueOrDefault("experiment_id")}");
        logger?.Invoke($"   Stage: {parameters.GetValueOrDefault("stage", "production")}");

        // Placeholder: In real implementation, promote model
        return Task.FromResult(new Dictionary<string, object>
        {
            ["promoted"] = true
        });
    }
}
