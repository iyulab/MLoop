namespace MLoop.CLI.Infrastructure.Configuration;

/// <summary>
/// MLoop project configuration
/// </summary>
public class MLoopConfig
{
    public string? ProjectName { get; set; }
    public string? Task { get; set; }
    public string? LabelColumn { get; set; }
    public TrainingSettings? Training { get; set; }
    public DataSettings? Data { get; set; }
    public ModelSettings? Model { get; set; }
}

/// <summary>
/// Training configuration settings
/// </summary>
public class TrainingSettings
{
    public int TimeLimitSeconds { get; set; } = 300;
    public string Metric { get; set; } = "accuracy";
    public double TestSplit { get; set; } = 0.2;
}

/// <summary>
/// Data path settings
/// </summary>
public class DataSettings
{
    public string? Train { get; set; }
    public string? Test { get; set; }
}

/// <summary>
/// Model output settings
/// </summary>
public class ModelSettings
{
    public string OutputDir { get; set; } = "models/staging";
}
