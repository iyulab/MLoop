using Microsoft.ML;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.Core.Hooks;
using MLoop.Extensibility.Hooks;
using MLoop.Extensibility.Preprocessing;

namespace MLoop.CLI.Infrastructure.ML;

/// <summary>
/// Runs the hooks of the two stages a batch command owns: <c>pre-predict</c> before <c>mloop predict</c>
/// scores a table, <c>post-evaluate</c> after <c>mloop evaluate</c> has its metrics. Training runs the
/// other two (<see cref="TrainingEngine"/>). Every stage of <see cref="HookType"/> has exactly one command
/// that runs it — <c>HookStagesRunContractTests</c> holds that.
/// </summary>
internal static class BatchCommandHooks
{
    /// <summary>
    /// Runs the hooks of <paramref name="stage"/> and says which one refused, if any (<c>null</c> when the
    /// command may go on). A project with no hook in the stage's folder pays a directory check and nothing else.
    /// </summary>
    public static async Task<string?> RefusalAsync(
        HookType stage,
        string projectRoot,
        string modelName,
        string? experimentId,
        string? taskType,
        string? labelColumn,
        string dataFile,
        string modelPath,
        object? metrics = null)
    {
        var hooks = new HookEngine(projectRoot, new ConsoleLogger());
        if (!hooks.HasHooks(stage))
            return null;

        // The data is a file the hook reads (Metadata["DataFile"], a CSV: what the command goes on to read);
        // a hook that wants rows reads it, as the templates do.
        var metadata = new Dictionary<string, object>
        {
            ["DataFile"] = dataFile,
            ["ModelName"] = modelName,
            ["ModelPath"] = modelPath
        };
        if (experimentId is { Length: > 0 })
            metadata["ExperimentId"] = experimentId;
        if (taskType is { Length: > 0 })
            metadata["TaskType"] = taskType;
        if (labelColumn is { Length: > 0 })
            metadata["LabelColumn"] = labelColumn;

        var context = new HookContext
        {
            HookType = stage,
            HookName = string.Empty, // set by HookEngine
            MLContext = new MLContext(seed: 0),
            DataView = null,
            Model = null,
            ExperimentResult = null,
            Metrics = metrics,
            ProjectRoot = projectRoot,
            Logger = new ConsoleLogger(),
            Metadata = metadata
        };

        return await hooks.RefusalAsync(stage, context);
    }
}
