// Pre-train hook: refuse to train on a file that is too small, too imbalanced or too empty.
// Put it in .mloop/scripts/hooks/pre-train/ — the folder names the stage (pre-train, post-train, pre-predict, post-evaluate).
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using MLoop.Extensibility.Hooks;

public class DataValidationHook : IMLoopHook
{
    public string Name => "Data Validation";

    // ⚙️ Adjust for your domain
    private const int MinRows = 100;              // fewer rows than this: abort
    private const double MaxMissingShare = 0.30;  // a column more than 30% empty: warn
    private const double MaxMajorityShare = 0.95; // one class above 95% of the rows: warn

    public Task<HookResult> ExecuteAsync(HookContext ctx)
    {
        // The pre-train stage reads a file: its path is in the hook's metadata.
        var dataFile = ctx.GetMetadata<string>("DataFile", "");
        if (string.IsNullOrEmpty(dataFile) || !File.Exists(dataFile))
        {
            ctx.Logger.Warning("No data file to check");
            return Task.FromResult(HookResult.Continue());
        }

        // A small CSV reader is enough for a first check: split on commas (a quoted comma would shift columns —
        // use a real parser for such files).
        var lines = File.ReadLines(dataFile).Where(l => !string.IsNullOrWhiteSpace(l)).ToList();
        if (lines.Count == 0)
            return Task.FromResult(HookResult.Abort("The data file is empty"));

        var header = lines[0].Split(',');
        var rows = lines.Skip(1).Select(l => l.Split(',')).ToList();
        ctx.Logger.Info($"Dataset size: {rows.Count:N0} rows, {header.Length} columns");

        if (rows.Count < MinRows)
            return Task.FromResult(HookResult.Abort($"Insufficient data: {rows.Count} rows (at least {MinRows})"));

        // Missing values per column
        for (int c = 0; c < header.Length; c++)
        {
            var empty = rows.Count(r => c >= r.Length || string.IsNullOrWhiteSpace(r[c]));
            if ((double)empty / rows.Count > MaxMissingShare)
                ctx.Logger.Warning($"Column '{header[c].Trim()}' is {100.0 * empty / rows.Count:F0}% empty");
        }

        // Class balance of the label, when the run names one
        var label = ctx.GetMetadata<string>("LabelColumn", "");
        var labelIndex = Array.FindIndex(header, h => h.Trim().Equals(label, StringComparison.OrdinalIgnoreCase));
        if (labelIndex >= 0)
        {
            var biggest = rows.Where(r => labelIndex < r.Length)
                .GroupBy(r => r[labelIndex].Trim())
                .Select(g => g.Count())
                .DefaultIfEmpty(0)
                .Max();
            if ((double)biggest / rows.Count > MaxMajorityShare)
                ctx.Logger.Warning($"One class is {100.0 * biggest / rows.Count:F0}% of the rows — a model may learn to answer it every time");
        }

        ctx.Logger.Info("Validation passed");
        return Task.FromResult(HookResult.Continue());
    }
}
