namespace MLoop.Core.Preprocessing;

/// <summary>
/// A single preprocessing step defined in mloop.yaml.
/// Each step maps to a FilePrepper DataPipeline operation.
/// </summary>
public class PrepStep
{
    /// <summary>
    /// Step type: fill-missing, normalize, remove-columns, rename-columns,
    /// drop-duplicates, extract-date, parse-datetime, filter-rows, scale
    /// </summary>
    public required string Type { get; set; }

    /// <summary>
    /// Target columns (for multi-column operations like fill-missing, normalize)
    /// </summary>
    public List<string>? Columns { get; set; }

    /// <summary>
    /// Target column (for single-column operations like extract-date, parse-datetime)
    /// </summary>
    public string? Column { get; set; }

    /// <summary>
    /// Method parameter (e.g., fill-missing: mean/median/mode/forward/backward/constant,
    /// normalize: min-max/z-score/robust, scale: min-max/z-score/robust)
    /// </summary>
    public string? Method { get; set; }

    /// <summary>
    /// Date features to extract (year, month, day, hour, minute, day_of_week, day_of_year, week_of_year, quarter)
    /// </summary>
    public List<string>? Features { get; set; }

    /// <summary>
    /// Whether to remove the original column after transformation
    /// </summary>
    public bool RemoveOriginal { get; set; }

    /// <summary>
    /// Key columns for deduplication
    /// </summary>
    public List<string>? KeyColumns { get; set; }

    /// <summary>
    /// Whether to keep first occurrence when deduplicating (default: true)
    /// </summary>
    public bool KeepFirst { get; set; } = true;

    /// <summary>
    /// DateTime format string for parse-datetime
    /// </summary>
    public string? Format { get; set; }

    /// <summary>
    /// Output column name for transformed data
    /// </summary>
    public string? OutputColumn { get; set; }

    /// <summary>
    /// Constant value for fill-missing with method: constant
    /// </summary>
    public string? Value { get; set; }

    /// <summary>
    /// Column mapping for rename-columns (old_name: new_name)
    /// </summary>
    public Dictionary<string, string>? Mapping { get; set; }

    /// <summary>
    /// Filter operator for filter-rows (gt, gte, lt, lte, eq, neq, contains, not-contains)
    /// </summary>
    public string? Operator { get; set; }

    /// <summary>
    /// Window size for rolling aggregation
    /// </summary>
    public int WindowSize { get; set; }

    /// <summary>
    /// Time column for resample operations
    /// </summary>
    public string? TimeColumn { get; set; }

    /// <summary>
    /// Window specification for resample (e.g., "5T", "1H", "1D")
    /// </summary>
    public string? Window { get; set; }

    /// <summary>
    /// Output suffix for rolling columns (default: "_rolling")
    /// </summary>
    public string? OutputSuffix { get; set; }

    /// <summary>
    /// Expression for add-column (simple expressions: concat, math, constant)
    /// </summary>
    public string? Expression { get; set; }
}
