using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Storage;

namespace MLoop.CLI.Infrastructure.Configuration;

/// <summary>What a command reads when the user names no file: the project's declared file, else the conventional one.</summary>
/// <param name="Path">The file, or <c>null</c> when there is none.</param>
/// <param name="DeclaredMissing">The <c>data.*</c> value of <c>mloop.yaml</c> that names no existing file, when one was declared and the conventional file stood in (or was missing too).</param>
public sealed record DefaultDataFile(string? Path, string? DeclaredMissing)
{
    /// <summary>The sentence a command says when it found no file: what <c>mloop.yaml</c> declared, if anything, and the conventional name.</summary>
    public string NotFoundMessage(DatasetRole role, string noun)
    {
        var conventional = $"{ProjectLayout.DatasetsDirectory}/{ProjectDataFiles.ConventionalFileName(role)}";
        return DeclaredMissing is null
            ? $"No {noun} specified and {conventional} not found."
            : $"No {noun} specified: data.{ProjectDataFiles.KeyOf(role)} in mloop.yaml names '{DeclaredMissing}', which does not exist, and {conventional} not found.";
    }
}

/// <summary>
/// The file a command reads by default for a <see cref="DatasetRole"/> — the one place that knows the order:
/// <c>data.train</c> / <c>data.test</c> / <c>data.predict</c> of <c>mloop.yaml</c> when it names an existing file, else
/// <c>datasets/train.csv</c> / <c>test.csv</c> / <c>predict.csv</c>.
/// </summary>
/// <remarks>
/// <c>mloop init</c> writes <c>data.train</c> and <c>data.test</c> as the project's "default data paths", but only
/// <c>train</c> and <c>analyze</c> read the first and nothing read the second: <c>data.test: holdout.csv</c> left
/// <c>mloop evaluate</c> reading <c>datasets/test.csv</c> under the line <c>Auto-detected</c>, and <c>data.predict</c> was
/// never read. The same commands also found their conventional file only when <c>datasets/train.csv</c> existed, so a
/// project kept for scoring and predicting — its training data elsewhere — was told <c>datasets/test.csv</c> was missing
/// while it sat there. A declared file that does not exist falls back to the conventional one, as it always did for
/// training, and is named when neither exists (<c>mloop validate</c> reports it as well).
/// </remarks>
public static class ProjectDataFiles
{
    /// <summary>The default file for <paramref name="role"/> in the project at <paramref name="projectRoot"/>.</summary>
    public static DefaultDataFile Resolve(DatasetRole role, MLoopConfig? config, string projectRoot, IDatasetDiscovery discovery)
    {
        ArgumentNullException.ThrowIfNull(discovery);

        var declared = role switch
        {
            DatasetRole.Train => config?.Data?.Train,
            DatasetRole.Test => config?.Data?.Test,
            DatasetRole.Predict => config?.Data?.Predict,
            _ => throw new ArgumentOutOfRangeException(nameof(role), role, null),
        };

        string? declaredMissing = null;
        if (!string.IsNullOrWhiteSpace(declared))
        {
            var path = ProjectPaths.Resolve(declared, projectRoot);
            if (File.Exists(path))
                return new DefaultDataFile(path, null);
            declaredMissing = declared;
        }

        var conventional = discovery.FindFile(projectRoot, role);
        return new DefaultDataFile(conventional, conventional is null ? declaredMissing : null);
    }

    /// <summary>The key under <c>data:</c> in <c>mloop.yaml</c> that declares the file of <paramref name="role"/>.</summary>
    public static string KeyOf(DatasetRole role) => role switch
    {
        DatasetRole.Train => "train",
        DatasetRole.Test => "test",
        DatasetRole.Predict => "predict",
        _ => throw new ArgumentOutOfRangeException(nameof(role), role, null),
    };

    /// <summary>The name of the conventional file of <paramref name="role"/> in <c>datasets/</c>.</summary>
    public static string ConventionalFileName(DatasetRole role) => role switch
    {
        DatasetRole.Train => ProjectLayout.TrainFileName,
        DatasetRole.Test => ProjectLayout.TestFileName,
        DatasetRole.Predict => ProjectLayout.PredictFileName,
        _ => throw new ArgumentOutOfRangeException(nameof(role), role, null),
    };
}
