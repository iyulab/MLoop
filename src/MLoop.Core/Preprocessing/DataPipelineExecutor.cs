using FilePrepper.Pipeline;
using FilePrepper.Tasks.NormalizeData;
using MLoop.Extensibility.Preprocessing;

namespace MLoop.Core.Preprocessing;

/// <summary>
/// Executes a sequence of PrepSteps using FilePrepper's DataPipeline API.
/// Transforms CSV data through a fluent pipeline without intermediate file I/O.
/// </summary>
public class DataPipelineExecutor
{
    private readonly ILogger _logger;

    public DataPipelineExecutor(ILogger logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Executes preprocessing steps defined in mloop.yaml on the input CSV file.
    /// Returns the path to the preprocessed output file.
    /// </summary>
    public async Task<string> ExecuteAsync(
        string inputPath,
        List<PrepStep> steps,
        string outputPath)
    {
        _logger.Info($"Running {steps.Count} preprocessing step(s) via DataPipeline...");

        var pipeline = await DataPipeline.FromCsvAsync(inputPath);
        var originalRowCount = pipeline.RowCount;

        for (int i = 0; i < steps.Count; i++)
        {
            var step = steps[i];
            _logger.Info($"  [{i + 1}/{steps.Count}] {step.Type}");

            pipeline = ApplyStep(pipeline, step);
        }

        await pipeline.ToCsvAsync(outputPath);

        _logger.Info($"Preprocessing complete: {originalRowCount} rows -> {pipeline.RowCount} rows");
        return outputPath;
    }

    private DataPipeline ApplyStep(DataPipeline pipeline, PrepStep step)
    {
        return step.Type.ToLowerInvariant() switch
        {
            "fill-missing" or "fill_missing" => ApplyFillMissing(pipeline, step),
            "normalize" => ApplyNormalize(pipeline, step),
            "scale" => ApplyScale(pipeline, step),
            "remove-columns" or "remove_columns" => ApplyRemoveColumns(pipeline, step),
            "rename-columns" or "rename_columns" => ApplyRenameColumns(pipeline, step),
            "drop-duplicates" or "drop_duplicates" => ApplyDropDuplicates(pipeline, step),
            "extract-date" or "extract_date" => ApplyExtractDate(pipeline, step),
            "parse-datetime" or "parse_datetime" => ApplyParseDatetime(pipeline, step),
            "filter-rows" or "filter_rows" => ApplyFilterRows(pipeline, step),
            _ => throw new InvalidOperationException($"Unknown prep step type: '{step.Type}'")
        };
    }

    private DataPipeline ApplyFillMissing(DataPipeline pipeline, PrepStep step)
    {
        var columns = ResolveColumns(step);
        var method = ParseFillMethod(step.Method ?? "mean");

        pipeline.FillMissing(columns, method, step.Value);
        return pipeline;
    }

    private DataPipeline ApplyNormalize(DataPipeline pipeline, PrepStep step)
    {
        var columns = ResolveColumns(step);
        var method = ParseNormalizationMethod(step.Method ?? "min-max");

        pipeline.Normalize(columns, method);
        return pipeline;
    }

    private DataPipeline ApplyScale(DataPipeline pipeline, PrepStep step)
    {
        // Scale is an alias for normalize
        return ApplyNormalize(pipeline, step);
    }

    private DataPipeline ApplyRemoveColumns(DataPipeline pipeline, PrepStep step)
    {
        var columns = ResolveColumns(step);
        pipeline.RemoveColumns(columns);
        return pipeline;
    }

    private DataPipeline ApplyRenameColumns(DataPipeline pipeline, PrepStep step)
    {
        if (step.Mapping == null || step.Mapping.Count == 0)
        {
            throw new InvalidOperationException("rename-columns step requires 'mapping' parameter");
        }

        foreach (var (oldName, newName) in step.Mapping)
        {
            pipeline.RenameColumn(oldName, newName);
        }
        return pipeline;
    }

    private static DataPipeline ApplyDropDuplicates(DataPipeline pipeline, PrepStep step)
    {
        var keyColumns = step.KeyColumns?.ToArray();
        return pipeline.DropDuplicates(keyColumns, step.KeepFirst);
    }

    private DataPipeline ApplyExtractDate(DataPipeline pipeline, PrepStep step)
    {
        var column = step.Column
            ?? throw new InvalidOperationException("extract-date step requires 'column' parameter");

        var features = ParseDateFeatures(step.Features);

        pipeline.ExtractDateFeatures(column, features, step.RemoveOriginal);
        return pipeline;
    }

    private DataPipeline ApplyParseDatetime(DataPipeline pipeline, PrepStep step)
    {
        var column = step.Column
            ?? throw new InvalidOperationException("parse-datetime step requires 'column' parameter");

        var format = step.Format
            ?? throw new InvalidOperationException("parse-datetime step requires 'format' parameter");

        pipeline.ParseDateTime(column, format);
        return pipeline;
    }

    private DataPipeline ApplyFilterRows(DataPipeline pipeline, PrepStep step)
    {
        var column = step.Column
            ?? throw new InvalidOperationException("filter-rows step requires 'column' parameter");
        var op = step.Operator
            ?? throw new InvalidOperationException("filter-rows step requires 'operator' parameter");
        var value = step.Value
            ?? throw new InvalidOperationException("filter-rows step requires 'value' parameter");

        Func<Dictionary<string, string>, bool> predicate = op.ToLowerInvariant() switch
        {
            "gt" or ">" => row => double.TryParse(row.GetValueOrDefault(column), out var v) && v > double.Parse(value),
            "gte" or ">=" => row => double.TryParse(row.GetValueOrDefault(column), out var v) && v >= double.Parse(value),
            "lt" or "<" => row => double.TryParse(row.GetValueOrDefault(column), out var v) && v < double.Parse(value),
            "lte" or "<=" => row => double.TryParse(row.GetValueOrDefault(column), out var v) && v <= double.Parse(value),
            "eq" or "==" => row => row.GetValueOrDefault(column) == value,
            "neq" or "!=" => row => row.GetValueOrDefault(column) != value,
            "contains" => row => row.GetValueOrDefault(column)?.Contains(value, StringComparison.OrdinalIgnoreCase) == true,
            "not-contains" or "not_contains" => row => row.GetValueOrDefault(column)?.Contains(value, StringComparison.OrdinalIgnoreCase) != true,
            _ => throw new InvalidOperationException($"Unknown filter operator: '{op}'")
        };

        return pipeline.FilterRows(predicate);
    }

    private string[] ResolveColumns(PrepStep step)
    {
        if (step.Columns is { Count: > 0 })
            return step.Columns.ToArray();

        if (!string.IsNullOrEmpty(step.Column))
            return [step.Column];

        throw new InvalidOperationException(
            $"Step '{step.Type}' requires 'columns' (list) or 'column' (single) parameter");
    }

    private static FillMethod ParseFillMethod(string method)
    {
        return method.ToLowerInvariant() switch
        {
            "mean" or "average" => FillMethod.Mean,
            "median" => FillMethod.Median,
            "mode" => FillMethod.Mode,
            "forward" or "ffill" => FillMethod.Forward,
            "backward" or "bfill" => FillMethod.Backward,
            "constant" => FillMethod.Constant,
            _ => throw new InvalidOperationException($"Unknown fill method: '{method}'")
        };
    }

    private static NormalizationMethod ParseNormalizationMethod(string method)
    {
        return method.ToLowerInvariant() switch
        {
            "min-max" or "min_max" or "minmax" => NormalizationMethod.MinMax,
            "z-score" or "z_score" or "zscore" or "standard" => NormalizationMethod.ZScore,
            "robust" => NormalizationMethod.Robust,
            _ => throw new InvalidOperationException($"Unknown normalization method: '{method}'")
        };
    }

    private static DateFeatures ParseDateFeatures(List<string>? features)
    {
        if (features == null || features.Count == 0)
            return DateFeatures.All;

        DateFeatures result = 0;
        foreach (var feature in features)
        {
            result |= feature.ToLowerInvariant() switch
            {
                "year" => DateFeatures.Year,
                "month" => DateFeatures.Month,
                "day" => DateFeatures.Day,
                "hour" => DateFeatures.Hour,
                "minute" => DateFeatures.Minute,
                "day_of_week" or "dayofweek" or "day-of-week" => DateFeatures.DayOfWeek,
                "day_of_year" or "dayofyear" or "day-of-year" => DateFeatures.DayOfYear,
                "week_of_year" or "weekofyear" or "week-of-year" => DateFeatures.WeekOfYear,
                "quarter" => DateFeatures.Quarter,
                "all" => DateFeatures.All,
                _ => throw new InvalidOperationException($"Unknown date feature: '{feature}'")
            };
        }
        return result;
    }
}
