using MLoop.CLI.Commands;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
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

        var handler = new LabelValueHandler(new CsvHelperImpl(), SilentLogger.Instance);
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

    /// <summary>The refusal when the training data holds one class — no classifier can be trained on it.</summary>
    public const string SingleClassRefusal = "Cannot train a classifier with only one class.";

    /// <summary>What to check when <see cref="SingleClassRefusal"/> is the answer.</summary>
    public const string SingleClassGuidance =
        "Check if the correct label column is specified, or if the data contains only one category.";

    /// <summary>
    /// Why <paramref name="balance"/> cannot be honoured for <paramref name="definition"/>, or null: not a balance
    /// option, a task that is not classification, or a weight column that already decides how often rows count. An
    /// option the run silently ignored would train a model its user believes balanced.
    /// </summary>
    public static string? BalanceConflict(ModelDefinition definition, string? balance)
    {
        ArgumentNullException.ThrowIfNull(definition);
        if (DataBalancer.OptionError(balance) is { } optionError)
            return optionError;
        if (string.IsNullOrEmpty(balance) || balance.Equals("none", StringComparison.OrdinalIgnoreCase))
            return null;
        if (!AutoMLRunner.IsClassification(definition.Task))
            return $"--balance applies to classification tasks; '{definition.Task}' has no classes to balance.";
        return ModelTrainingConfig.WeightConflict(definition, balance);
    }

    /// <summary>The files a run trains and tests on once its classes are balanced.</summary>
    /// <param name="TrainFile">The (balanced, when balancing applied) training rows.</param>
    /// <param name="TestFile">The held-out rows, split off before balancing; null without a test split.</param>
    /// <param name="Split">The stratified split, when there was one.</param>
    /// <param name="Result">What balancing did.</param>
    public sealed record Balanced(string TrainFile, string? TestFile, CsvSplitter.SplitResult? Split, DataBalancer.BalanceResult Result);

    /// <summary>
    /// Balances <paramref name="dataFile"/> by <paramref name="balance"/> for <c>mloop train</c> and <c>POST /train</c>
    /// alike. With a test split the rows are split (stratified) <b>before</b> balancing and only the train part is
    /// balanced — replicated minority rows in the test part would score the model on copies of what it trained on.
    /// </summary>
    public static Balanced Balance(string dataFile, string label, string balance, double testSplit)
    {
        var balancer = new DataBalancer();
        if (testSplit <= 0)
        {
            var whole = balancer.Balance(dataFile, label, balance);
            return new Balanced(whole is { Applied: true, BalancedFilePath: { } path } ? path : dataFile, null, null, whole);
        }

        var split = new CsvSplitter().StratifiedSplit(dataFile, label, testSplit);
        var result = balancer.Balance(split.TrainFile, label, balance);
        return new Balanced(result is { Applied: true, BalancedFilePath: { } balanced } ? balanced : split.TrainFile,
            split.TestFile, split, result);
    }

    /// <summary>The class distribution of <paramref name="dataFile"/>'s <paramref name="label"/> — what both entry points warn on.</summary>
    public static Task<MLoop.Core.Diagnostics.ClassDistributionResult> ClassesAsync(string dataFile, string label)
        => new MLoop.Core.Diagnostics.ClassDistributionAnalyzer(new CsvHelperImpl()).AnalyzeAsync(dataFile, label);
}
