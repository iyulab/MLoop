using MLoop.Extensibility;
using Microsoft.ML;

/// <summary>
/// Pre-train hook that validates data quality before training.
/// Checks for: minimum rows, class imbalance, missing values
/// </summary>
public class DataValidationHook : IMLoopHook
{
    public string Name => "Data Quality Check";

    public async Task<HookResult> ExecuteAsync(HookContext ctx)
    {
        try
        {
            var df = ctx.DataView;
            var preview = df.Preview(maxRows: 1000);  // Sample for performance

            // 1. Minimum row count validation
            var rowCount = preview.RowView.Length;
            ctx.Logger.Info($"📊 Validating {rowCount} rows...");

            if (rowCount < 100)
            {
                ctx.Logger.Error($"❌ Insufficient data: {rowCount} rows");
                return HookResult.Abort(
                    $"Training requires at least 100 rows, found {rowCount}. " +
                    "Please collect more data.");
            }

            // 2. Class imbalance check
            var labelCol = ctx.Metadata["LabelColumn"] as string;
            if (!string.IsNullOrEmpty(labelCol))
            {
                var distribution = AnalyzeClassBalance(df, labelCol, ctx.MLContext);

                if (distribution.ImbalanceRatio > 20)
                {
                    ctx.Logger.Warning(
                        $"⚠️  Severe class imbalance detected: " +
                        $"{distribution.ImbalanceRatio:F1}:1 " +
                        $"({distribution.MajorityClass}:{distribution.MinorityClass})");
                    ctx.Logger.Warning(
                        "   Consider: SMOTE, class weights, or collecting more minority samples");
                }
                else if (distribution.ImbalanceRatio > 5)
                {
                    ctx.Logger.Info(
                        $"⚠️  Moderate class imbalance: {distribution.ImbalanceRatio:F1}:1");
                }
            }

            // 3. Missing value analysis
            var missingStats = AnalyzeMissingValues(preview);
            if (missingStats.HasProblematicColumns)
            {
                ctx.Logger.Warning("⚠️  Columns with high missing value rates:");
                foreach (var col in missingStats.ProblematicColumns)
                {
                    ctx.Logger.Warning(
                        $"   - {col.Name}: {col.MissingPercent:F1}% missing");
                }
                ctx.Logger.Warning(
                    "   ML.NET will handle missing values, but consider imputation");
            }

            // Success
            ctx.Logger.Info($"✅ Data quality check passed: {rowCount} rows");
            return HookResult.Continue();
        }
        catch (Exception ex)
        {
            ctx.Logger.Error($"❌ Data validation failed: {ex.Message}");
            // Don't fail training on validation errors
            return HookResult.Continue();
        }
    }

    private ClassDistribution AnalyzeClassBalance(
        IDataView dataView,
        string labelColumn,
        MLContext mlContext)
    {
        // Count label values
        var labelData = mlContext.Data.CreateEnumerable<LabelData>(
            dataView,
            reuseRowObject: false
        ).ToList();

        var labelCounts = labelData
            .GroupBy(x => x.Label)
            .Select(g => new { Label = g.Key, Count = g.Count() })
            .OrderByDescending(x => x.Count)
            .ToList();

        if (labelCounts.Count < 2)
        {
            return new ClassDistribution
            {
                ImbalanceRatio = 1.0,
                MajorityClass = labelCounts[0].Count,
                MinorityClass = labelCounts[0].Count
            };
        }

        var majority = labelCounts[0].Count;
        var minority = labelCounts[labelCounts.Count - 1].Count;

        return new ClassDistribution
        {
            ImbalanceRatio = (double)majority / minority,
            MajorityClass = majority,
            MinorityClass = minority
        };
    }

    private MissingValueStats AnalyzeMissingValues(DataDebuggerPreview preview)
    {
        var problematicColumns = new List<ColumnMissingInfo>();
        var schema = preview.Schema;

        for (int i = 0; i < schema.Count; i++)
        {
            var column = schema[i];
            int missingCount = 0;
            int totalCount = preview.RowView.Length;

            foreach (var row in preview.RowView)
            {
                var value = row.Values[i].Value;
                if (value == null ||
                    (value is float f && float.IsNaN(f)) ||
                    (value is double d && double.IsNaN(d)))
                {
                    missingCount++;
                }
            }

            double missingPercent = (double)missingCount / totalCount * 100;

            // Flag columns with >30% missing values
            if (missingPercent > 30)
            {
                problematicColumns.Add(new ColumnMissingInfo
                {
                    Name = column.Name,
                    MissingPercent = missingPercent,
                    MissingCount = missingCount,
                    TotalCount = totalCount
                });
            }
        }

        return new MissingValueStats
        {
            ProblematicColumns = problematicColumns,
            HasProblematicColumns = problematicColumns.Any()
        };
    }

    // Helper classes
    private class LabelData
    {
        public bool Label { get; set; }
    }

    private class ClassDistribution
    {
        public double ImbalanceRatio { get; set; }
        public int MajorityClass { get; set; }
        public int MinorityClass { get; set; }
    }

    private class MissingValueStats
    {
        public List<ColumnMissingInfo> ProblematicColumns { get; set; } = new();
        public bool HasProblematicColumns { get; set; }
    }

    private class ColumnMissingInfo
    {
        public string Name { get; set; } = string.Empty;
        public double MissingPercent { get; set; }
        public int MissingCount { get; set; }
        public int TotalCount { get; set; }
    }
}
