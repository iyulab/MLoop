namespace MLoop.Extensibility.Metrics;

/// <summary>
/// Result of custom business metric calculation.
/// </summary>
public class MetricResult
{
    /// <summary>
    /// Metric name for identification.
    /// Example: "Expected Profit", "Customer Lifetime Value"
    /// </summary>
    public required string Name { get; init; }

    /// <summary>
    /// Calculated metric value.
    /// Interpretation depends on metric type (profit, cost, ROI, etc.)
    /// </summary>
    public required double Value { get; init; }

    /// <summary>
    /// Human-readable description of the result.
    /// Example: "Profit from 150 TPs - 20 FPs = $14,600"
    /// </summary>
    public string? Description { get; init; }

    /// <summary>
    /// Additional details or breakdown of the calculation.
    /// Optional: Can include sub-metrics, confidence intervals, etc.
    /// </summary>
    public Dictionary<string, object>? Details { get; init; }

    /// <summary>
    /// Returns string representation for logging and display.
    /// </summary>
    public override string ToString()
    {
        var result = $"{Name}: {Value:F2}";
        if (!string.IsNullOrEmpty(Description))
        {
            result += $" ({Description})";
        }
        return result;
    }
}
