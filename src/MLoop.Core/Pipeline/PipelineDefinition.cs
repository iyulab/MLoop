namespace MLoop.Core.Pipeline;

/// <summary>
/// Pipeline definition for ML workflows
/// </summary>
public class PipelineDefinition
{
    public required string Name { get; init; }
    public string? Description { get; init; }
    public required List<PipelineStep> Steps { get; init; }
    public Dictionary<string, object>? Variables { get; init; }
}

/// <summary>
/// Individual pipeline step
/// </summary>
public class PipelineStep
{
    public required string Name { get; init; }
    public required string Type { get; init; }  // preprocess, train, evaluate, predict, promote
    public required Dictionary<string, object> Parameters { get; init; }
    public bool ContinueOnError { get; init; }
    public List<string>? DependsOn { get; init; }  // List of step names this step depends on
    public bool Parallel { get; init; }  // Can this step run in parallel with others?
    public StepCondition? Condition { get; init; }  // Optional condition for execution
}

/// <summary>
/// Condition for step execution
/// </summary>
public class StepCondition
{
    public required string Expression { get; init; }  // e.g., "$train_step.metric > 0.9"
    public ConditionOperator Operator { get; init; }
    public string? Value { get; init; }
}

/// <summary>
/// Condition operators for step execution
/// </summary>
public enum ConditionOperator
{
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Contains,
    NotContains
}

/// <summary>
/// Pipeline execution result
/// </summary>
public class PipelineResult
{
    public required string PipelineName { get; init; }
    public required DateTime StartTime { get; init; }
    public required DateTime EndTime { get; init; }
    public required TimeSpan Duration { get; init; }
    public required PipelineStatus Status { get; init; }
    public required List<StepResult> StepResults { get; init; }
    public string? Error { get; init; }
}

/// <summary>
/// Individual step result
/// </summary>
public class StepResult
{
    public required string StepName { get; init; }
    public required string StepType { get; init; }
    public required DateTime StartTime { get; init; }
    public required DateTime EndTime { get; init; }
    public required TimeSpan Duration { get; init; }
    public required StepStatus Status { get; init; }
    public Dictionary<string, object>? Outputs { get; init; }
    public string? Error { get; init; }
}

/// <summary>
/// Pipeline execution status
/// </summary>
public enum PipelineStatus
{
    Running,
    Completed,
    Failed,
    PartiallyCompleted
}

/// <summary>
/// Step execution status
/// </summary>
public enum StepStatus
{
    Pending,
    Running,
    Completed,
    Failed,
    Skipped
}
