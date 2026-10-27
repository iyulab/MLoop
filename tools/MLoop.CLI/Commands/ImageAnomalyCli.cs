using System.Globalization;
using System.Text.Json;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.Core.Embedding;
using MLoop.Core.ImageAnomaly;
using MLoop.Core.Storage;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// What <c>mloop predict</c> and <c>mloop evaluate</c> do for an image-anomaly model: its input is pictures, not rows, so it
/// bypasses the table paths of both and scores each image against the bank beside the model.
/// </summary>
internal static class ImageAnomalyCli
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull,
    };

    /// <summary>The images a path names: one image, or every image under a folder.</summary>
    internal static IReadOnlyList<string> ImagesOf(string path) =>
        File.Exists(path)
            ? ImageAnomalyTrainer.ListImages(Path.GetDirectoryName(Path.GetFullPath(path))!)
                .Where(f => string.Equals(f, Path.GetFullPath(path), StringComparison.Ordinal)).ToList()
            : ImageAnomalyTrainer.ListImages(path);

    private static float Round(float value) => MathF.Round(value, 5);

    /// <summary><c>mloop predict &lt;images&gt;</c> for an image-anomaly model.</summary>
    internal static async Task<int> PredictAsync(
        string projectRoot, string modelPath, string? experimentId, string? dataFile, string? output, string modelName,
        bool json, FileSystemManager fileSystem)
    {
        if (string.IsNullOrWhiteSpace(dataFile))
        {
            ErrorConsole.Error("An image-anomaly model scores pictures: name an image or a folder of them.",
                "mloop predict <image-or-folder>");
            return 1;
        }
        var path = ProjectPaths.Resolve(dataFile, projectRoot);
        if (!Directory.Exists(path) && !File.Exists(path))
        {
            ErrorConsole.Error($"Data not found: {path}", ErrorConsole.PathNotFoundTip(projectRoot));
            return 1;
        }
        var images = ImagesOf(path);
        if (images.Count == 0)
        {
            ErrorConsole.Error($"No images found in {Path.GetRelativePath(projectRoot, path)}.",
                $"An image-anomaly model scores image files ({ImageAnomalyTrainer.ExtensionList}).");
            return 1;
        }

        var (scorer, refusal) = ImageAnomalyScorer.Open(modelPath);
        if (scorer is null)
        {
            ErrorConsole.Error(Markup.Escape(refusal!));
            return 1;
        }

        IReadOnlyList<ImageScore> scores;
        try
        {
            scores = await Task.Run(() => scorer.Score([.. images.Select(File.ReadAllBytes)]));
        }
        catch (UnreadableImageException ex)
        {
            ErrorConsole.Error(Markup.Escape($"'{images[ex.Index]}' could not be read as an image: {ex.Message}"));
            return 1;
        }

        var grid = scorer.Model.Grid;
        var document = new
        {
            Model = modelName,
            ExperimentId = experimentId,
            Encoder = new { scorer.Model.Encoder.Id, scorer.Model.Encoder.Sha256, scorer.Model.Encoder.Preprocessor },
            Grid = new { grid.Rows, grid.Columns, grid.PatchSize, grid.CropSize },
            Baseline = scorer.Model.Baseline,
            Images = images.Select((file, i) => new
            {
                File = Path.GetRelativePath(projectRoot, file),
                scores[i].Score,
                Crop = new { X0 = scores[i].Crop.X0, Y0 = scores[i].Crop.Y0, X1 = scores[i].Crop.X1, Y1 = scores[i].Crop.Y1 },
                Distances = scores[i].Distances.Select(Round).ToArray(),
            }).ToArray(),
            Warnings = JsonOutputScope.Warnings is { Count: > 0 } w ? w : null,
        };
        var text = JsonSerializer.Serialize(document, JsonOptions);

        if (json)
        {
            Console.WriteLine(text);
            return 0;
        }

        string outputPath;
        if (string.IsNullOrEmpty(output))
        {
            var dir = fileSystem.CombinePath(projectRoot, ProjectLayout.PredictionsDirectory);
            await fileSystem.CreateDirectoryAsync(dir, CancellationToken.None);
            outputPath = fileSystem.CombinePath(dir,
                ProjectLayout.PredictionFileName(modelName, ProjectLayout.PredictionKind.Scores, DateTimeOffset.Now));
        }
        else
        {
            outputPath = ProjectPaths.Resolve(output, projectRoot);
            var outputDir = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrEmpty(outputDir) && !Directory.Exists(outputDir))
                await fileSystem.CreateDirectoryAsync(outputDir, CancellationToken.None);
        }
        await File.WriteAllTextAsync(outputPath, text, new System.Text.UTF8Encoding(false));

        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[green]Scores Complete![/]").LeftJustified());
        AnsiConsole.MarkupLine($"[green]>[/] Model: [cyan]{Markup.Escape(modelName)}[/]  encoder [cyan]{Markup.Escape(scorer.Model.Encoder.Id)}[/]");
        if (scorer.Model.Baseline is { } baseline)
            AnsiConsole.MarkupLine(string.Create(CultureInfo.InvariantCulture,
                $"[green]>[/] Held-out normal images scored: median {baseline.Median:0.0000} · p95 {baseline.P95:0.0000} · max {baseline.Max:0.0000} ({baseline.Images} images)"));
        var table = new Table().Border(TableBorder.Rounded).AddColumn("Image").AddColumn(new TableColumn("Score").RightAligned()).AddColumn("Hottest patch (row, col)");
        foreach (var (file, i) in images.Select((f, i) => (f, i)).OrderByDescending(p => scores[p.i].Score).Take(15))
        {
            var hottest = Array.IndexOf(scores[i].Distances, scores[i].Distances.Max());
            table.AddRow(Markup.Escape(Path.GetRelativePath(projectRoot, file)),
                scores[i].Score.ToString("0.0000", CultureInfo.InvariantCulture), $"{hottest / grid.Columns}, {hottest % grid.Columns}");
        }
        AnsiConsole.Write(table);
        if (images.Count > 15)
            AnsiConsole.MarkupLine($"[grey]The 15 highest of {images.Count:N0} images; all are in the file.[/]");
        ValueLine.Write("[green]>[/] Output saved to: ", Path.GetRelativePath(projectRoot, outputPath));
        return 0;
    }

    /// <summary>
    /// <c>mloop evaluate &lt;folder&gt;</c> for an image-anomaly model: the folder's subfolders are the classes — one of normal
    /// images (named <c>normal</c>, <c>good</c> or <c>ok</c>, or <paramref name="normalClass"/>), the others anomalous.
    /// </summary>
    internal static async Task<int> EvaluateAsync(
        string projectRoot, string modelPath, string modelName, string? experimentId, string? testData, string? normalClass, bool json)
    {
        var folder = testData is null ? Path.Combine(projectRoot, ProjectLayout.DatasetsDirectory, "test") : ProjectPaths.Resolve(testData, projectRoot);
        if (!Directory.Exists(folder))
        {
            ErrorConsole.Error($"Test folder not found: {folder}",
                "Give a folder with one subfolder of normal images (normal, good or ok) and one or more of anomalous ones: mloop evaluate <exp-id> <folder>");
            return 1;
        }

        var normalNames = normalClass is { Length: > 0 } ? [normalClass.Trim()] : ImageAnomalyEvaluator.NormalFolderNames;
        var classes = Directory.GetDirectories(folder).OrderBy(d => d, StringComparer.Ordinal).ToList();
        var normalDirs = classes.Where(d => normalNames.Contains(Path.GetFileName(d), StringComparer.OrdinalIgnoreCase)).ToList();
        var anomalyDirs = classes.Except(normalDirs).ToList();
        if (normalDirs.Count == 0 || anomalyDirs.Count == 0)
        {
            ErrorConsole.Error(Markup.Escape(
                    $"{Path.GetFileName(folder)} has {classes.Count} subfolder(s): {string.Join(", ", classes.Select(Path.GetFileName))}."),
                $"Evaluation tells normal images from anomalous ones: one subfolder named {string.Join(" / ", normalNames)} (normal) and at least one other (anomalous). --normal-class names the normal one.");
            return 1;
        }

        var (scorer, refusal) = ImageAnomalyScorer.Open(modelPath);
        if (scorer is null)
        {
            ErrorConsole.Error(Markup.Escape(refusal!));
            return 1;
        }

        double[] ScoreAll(IEnumerable<string> dirs)
        {
            var files = dirs.SelectMany(ImageAnomalyTrainer.ListImages).ToList();
            try
            {
                return [.. scorer.Score([.. files.Select(File.ReadAllBytes)]).Select(s => s.Score)];
            }
            catch (UnreadableImageException ex)
            {
                throw new InvalidOperationException($"'{files[ex.Index]}' could not be read as an image: {ex.Message}", ex);
            }
        }

        AnomalyQuality quality;
        try
        {
            var normal = await Task.Run(() => ScoreAll(normalDirs));
            var anomalous = await Task.Run(() => ScoreAll(anomalyDirs));
            quality = ImageAnomalyEvaluator.Evaluate(normal, anomalous);
        }
        catch (ArgumentException ex)
        {
            ErrorConsole.Error(Markup.Escape($"{ex.Message} ({Path.GetRelativePath(projectRoot, folder)})"));
            return 1;
        }
        catch (InvalidOperationException ex)
        {
            ErrorConsole.Error(Markup.Escape(ex.Message));
            return 1;
        }

        if (json)
        {
            Console.WriteLine(JsonSerializer.Serialize(new
            {
                Model = modelName,
                ExperimentId = experimentId,
                TestDataFile = Path.GetRelativePath(projectRoot, folder),
                NormalClass = normalDirs.Select(Path.GetFileName),
                AnomalyClasses = anomalyDirs.Select(Path.GetFileName),
                TestMetrics = quality.AsMetrics(),
                Warnings = JsonOutputScope.Warnings is { Count: > 0 } w ? w : null,
            }, JsonOptions));
            return 0;
        }

        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule("[green]Evaluation Results[/]").LeftJustified());
        var table = new Table().Border(TableBorder.Rounded).AddColumn("Metric").AddColumn(new TableColumn("Test").RightAligned());
        table.AddRow("auroc", quality.Auroc.ToString("0.0000", CultureInfo.InvariantCulture));
        table.AddRow("anomalous images missed at 5 % false alarms", $"{quality.MissedAt5} of {quality.Anomalies}");
        table.AddRow("anomalous images missed at 10 % false alarms", $"{quality.MissedAt10} of {quality.Anomalies}");
        table.AddRow("normal images", quality.Normals.ToString(CultureInfo.InvariantCulture));
        AnsiConsole.Write(table);
        AnsiConsole.MarkupLine($"[grey]normal: {Markup.Escape(string.Join(", ", normalDirs.Select(Path.GetFileName)))} · anomalous: {Markup.Escape(string.Join(", ", anomalyDirs.Select(Path.GetFileName)))}[/]");
        return 0;
    }
}
