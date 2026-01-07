// Example PreTrain Hook: Data Quality Validation
//
// Purpose:
//   Validates training data quality before AutoML training begins.
//   Checks minimum row count and class balance for classification tasks.
//
// Installation:
//   Copy to: .mloop/scripts/hooks/pre-train/01_data_validation.cs
//
// Configuration:
//   Customize MIN_ROWS and CLASS_BALANCE_THRESHOLD as needed.

using System.Threading.Tasks;
using Microsoft.ML;
using MLoop.Extensibility.Hooks;

public class DataValidationHook : IMLoopHook
{
    public string Name => "Data Quality Validation";

    // Configuration
    private const int MIN_ROWS = 100;
    private const double CLASS_BALANCE_THRESHOLD = 0.05; // 5% minimum for minority class

    public Task<HookResult> ExecuteAsync(HookContext ctx)
    {
        try
        {
            if (ctx.DataView == null)
            {
                return Task.FromResult(HookResult.Abort("No training data provided"));
            }

            // Check 1: Minimum row count
            var rowCount = GetRowCount(ctx.DataView);
            ctx.Logger.Info($"📊 Dataset size: {rowCount:N0} rows");

            if (rowCount < MIN_ROWS)
            {
                return Task.FromResult(HookResult.Abort(
                    $"Insufficient data: {rowCount} rows (minimum: {MIN_ROWS})"));
            }

            // Check 2: Class balance (for classification tasks)
            var taskType = ctx.GetMetadata<string>("TaskType", "");
            if (taskType.Contains("Classification", StringComparison.OrdinalIgnoreCase))
            {
                var labelColumn = ctx.GetMetadata<string>("LabelColumn", "Label");
                var classBalance = CheckClassBalance(ctx.MLContext, ctx.DataView, labelColumn);

                if (classBalance.MinorityRatio < CLASS_BALANCE_THRESHOLD)
                {
                    ctx.Logger.Warning(
                        $"⚠️ Class imbalance detected: {classBalance.MinorityRatio:P2} " +
                        $"(threshold: {CLASS_BALANCE_THRESHOLD:P2})");
                    ctx.Logger.Warning("   Consider oversampling or class weight adjustment");
                }
                else
                {
                    ctx.Logger.Info(
                        $"✅ Class balance OK: {classBalance.MinorityRatio:P2} minority class");
                }
            }

            ctx.Logger.Info("✅ Data quality validation passed");
            return Task.FromResult(HookResult.Continue());
        }
        catch (Exception ex)
        {
            ctx.Logger.Error($"Data validation failed: {ex.Message}");
            return Task.FromResult(HookResult.Abort($"Validation error: {ex.Message}"));
        }
    }

    private long GetRowCount(IDataView dataView)
    {
        // IDataView.GetRowCount() may return null for some data sources
        long? count = dataView.GetRowCount();
        if (count.HasValue)
        {
            return count.Value;
        }

        // Fallback: enumerate rows to count (may be slow for large datasets)
        long rowCount = 0;
        using (var cursor = dataView.GetRowCursor(dataView.Schema))
        {
            while (cursor.MoveNext())
            {
                rowCount++;
            }
        }
        return rowCount;
    }

    private (int MajorityCount, int MinorityCount, double MinorityRatio) CheckClassBalance(
        MLContext mlContext,
        IDataView dataView,
        string labelColumn)
    {
        // Get label column
        var labelColumnIndex = dataView.Schema.GetColumnOrNull(labelColumn)?.Index;
        if (!labelColumnIndex.HasValue)
        {
            throw new InvalidOperationException($"Label column '{labelColumn}' not found");
        }

        // Count class occurrences
        var classCounts = new Dictionary<string, int>();
        using (var cursor = dataView.GetRowCursor(dataView.Schema[labelColumn]))
        {
            var labelGetter = cursor.GetGetter<ReadOnlyMemory<char>>(dataView.Schema[labelColumn]);
            while (cursor.MoveNext())
            {
                ReadOnlyMemory<char> labelValue = default;
                labelGetter(ref labelValue);
                var label = labelValue.ToString();

                if (!classCounts.ContainsKey(label))
                {
                    classCounts[label] = 0;
                }
                classCounts[label]++;
            }
        }

        // Calculate balance
        if (classCounts.Count < 2)
        {
            throw new InvalidOperationException("Expected at least 2 classes for classification");
        }

        var sortedCounts = classCounts.Values.OrderByDescending(c => c).ToList();
        int majorityCount = sortedCounts[0];
        int minorityCount = sortedCounts[sortedCounts.Count - 1];
        double minorityRatio = (double)minorityCount / (majorityCount + minorityCount);

        return (majorityCount, minorityCount, minorityRatio);
    }
}
