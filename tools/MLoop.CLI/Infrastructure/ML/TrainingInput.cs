using MLoop.CLI.Commands;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.Core.AutoML;
using MLoop.Core.Data;
using MLoop.Extensibility.Preprocessing;

namespace MLoop.CLI.Infrastructure.ML;

/// <summary>
/// What a data file goes through before it is trained on, where both <c>mloop train</c> and
/// <c>POST /train</c> need the same answer.
/// </summary>
public static class TrainingInput
{
    /// <summary>
    /// Whether rows whose label is empty are dropped when the user has not said. A row without a target
    /// can neither teach a supervised model nor score it. This was the default for classification only,
    /// and a recommendation set whose unrated visits (22% of rows) stayed in trained and scored a model
    /// at R² 0.54 — 0.54 that was not there: with those rows dropped the same model scored −0.64, below
    /// the quality gate. A time series is the exception — its label is the series, and dropping a row
    /// would close up the time between its neighbours — as are unsupervised tasks and image directories.
    /// </summary>
    public static bool DropsMissingLabelsByDefault(string? task) =>
        AutoMLRunner.RequiresLabel(task)
        && !AutoMLRunner.IsTimeSeriesTask(task)
        && !DataLoaderFactory.IsDirectoryBased(task);

    /// <summary>The file a job trains on, and what preparing it did.</summary>
    /// <param name="DataFile">The prepared file.</param>
    /// <param name="DroppedMissingLabels">Rows dropped for an empty label.</param>
    public sealed record Prepared(string DataFile, int DroppedMissingLabels);

    /// <summary>
    /// Prepares <paramref name="dataFile"/> for a <c>POST /train</c> job as <c>mloop train</c> prepares its
    /// own: read as a table, multi-line fields flattened, sampled down to <paramref name="maxRows"/>, and
    /// rows without a label dropped where the task drops them. Before this, a job trained on the file as
    /// given — <c>maxRows</c> was recorded and never applied, and an unlabeled row (a logged prediction
    /// nobody gave feedback on) was trained on with an empty label.
    /// </summary>
    /// <exception cref="InvalidOperationException">The label column could not be cleaned.</exception>
    public static async Task<Prepared> PrepareAsync(
        string dataFile, ModelDefinition definition, int? maxRows, string projectRoot)
    {
        ArgumentNullException.ThrowIfNull(definition);
        var file = await TabularDataFile.AsCsvAsync(dataFile);
        file = CsvDataLoader.FlattenMultiLineQuotedFields(file);

        if (maxRows is > 0)
            file = await TrainDataValidator.ApplySamplingIfNeededAsync(
                file, maxRows.Value, definition.Task, definition.Label, explicitStrategy: null, seed: 42, projectRoot);

        if (!DropsMissingLabelsByDefault(definition.Task) || string.IsNullOrEmpty(definition.Label))
            return new Prepared(file, 0);

        var handler = new LabelValueHandler(new CsvHelperImpl(), new SilentLogger());
        var analysis = await handler.AnalyzeLabelColumnAsync(file, definition.Label);
        if (!string.IsNullOrEmpty(analysis.Error) || !analysis.HasMissingValues)
            return new Prepared(file, 0);

        var cleaned = Path.Combine(Path.GetDirectoryName(file)!,
            $"{Path.GetFileNameWithoutExtension(file)}_cleaned{Path.GetExtension(file)}");
        var result = await handler.DropMissingLabelsAsync(file, cleaned, definition.Label);
        if (!result.Success)
            throw new InvalidOperationException($"Failed to drop rows with an empty label '{definition.Label}': {result.Error}");
        return new Prepared(result.OutputPath!, result.DroppedRowCount);
    }

    private sealed class SilentLogger : ILogger
    {
        public void Debug(string message) { }
        public void Info(string message) { }
        public void Warning(string message) { }
        public void Error(string message) { }
        public void Error(string message, Exception exception) { }
    }
}
