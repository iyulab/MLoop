using MLoop.CLI.Commands;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.Core.Data;
using MLoop.Core.Models;
using MLoop.Core.Prediction;
using MLoop.Core.Storage;
using MLoop.Core.Training;
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
        TrainingRunner.RequiresLabel(task)
        && !TrainingRunner.IsTimeSeriesTask(task)
        && !DataLoaderFactory.IsDirectoryBased(task);

    /// <summary>
    /// Why a task that learns from a label column cannot train on <paramref name="dataFile"/>: the label is not one of its columns. Said
    /// first, and as that, with the columns the file has - the steps after it read the label, and one of them fails with "only one class"
    /// when there is none. <c>null</c> when the label is a column, or the task reads no label column from a table. <paramref name="reportedAs"/>
    /// is the file the user gave when <paramref name="dataFile"/> is a copy made of it (a workbook's converted table, a prepared file): the
    /// refusal names the user's file, not a scratch name.
    /// </summary>
    public static string? LabelColumnRefusal(string dataFile, string? task, string? label, string? reportedAs = null)
    {
        if (string.IsNullOrWhiteSpace(label) || !TrainingRunner.RequiresLabel(task) || DataLoaderFactory.IsDirectoryBased(task))
            return null;
        if (HeaderColumns(dataFile) is not { } columns)
            return null;
        if (columns.Any(column => string.Equals(column, label, StringComparison.Ordinal)))
            return null;
        var message = $"Label column '{label}' not found in {Path.GetFileName(reportedAs ?? dataFile)}. Available columns: {string.Join(", ", columns)}.";
        if (OtherDelimiter(columns) is { } other)
            return $"{message} The header reads as one column, and it looks separated by {other} - mloop reads a .csv file as comma-separated (a tab-separated one when it is named .tsv); save it with commas between the columns.";
        return columns.FirstOrDefault(column => string.Equals(column, label, StringComparison.OrdinalIgnoreCase)) is { } sameLetters
            ? $"{message} '{sameLetters}' differs only in case; column names are case-sensitive."
            : message;
    }

    /// <summary>
    /// The columns of a CSV/TSV file's header row, read in the file's own encoding: a Korean name against a CP949 file (the usual
    /// spreadsheet export) is no column when the header is read as UTF-8, and a refusal listed the columns as replacement characters.
    /// <c>null</c> for any other file, an unreadable one, or one with no header row.
    /// </summary>
    private static IReadOnlyList<string>? HeaderColumns(string dataFile)
    {
        if (!dataFile.EndsWith(".csv", StringComparison.OrdinalIgnoreCase) && !dataFile.EndsWith(".tsv", StringComparison.OrdinalIgnoreCase))
            return null;
        string? header;
        try
        {
            using var reader = EncodingDetector.OpenReader(dataFile);
            header = reader.ReadLine()?.TrimStart('\uFEFF');
        }
        catch (IOException) { return null; }
        return string.IsNullOrEmpty(header) ? null : CsvFieldParser.ParseFields(header);
    }

    /// <summary>
    /// A preprocessing step that is fitted inside training (<c>normalize</c>, <c>scale</c>, <c>fill-missing</c> by mean) names its
    /// columns in <c>mloop.yaml</c>, and the data has to have them. Before this, a name with no column in the data passed the
    /// preprocessing stage and the first thing to say so was the trainer search, failing every trainer three times — naming the column
    /// but not the step, after the setup of a whole run.
    /// </summary>
    /// <param name="dataFile">The file training reads, after the steps that change the table have run.</param>
    /// <param name="stepColumns">Every column the in-training steps name.</param>
    public static string? PrepColumnsRefusal(string dataFile, IReadOnlyList<string> stepColumns, string? reportedAs = null)
    {
        if (stepColumns.Count == 0 || HeaderColumns(dataFile) is not { } columns)
            return null;
        var absent = stepColumns.Where(name => !columns.Contains(name, StringComparer.Ordinal)).Distinct(StringComparer.Ordinal).ToList();
        if (absent.Count == 0)
            return null;
        var named = string.Join(", ", absent.Select(name => $"'{name}'"));
        var message = $"A prep step in mloop.yaml names {(absent.Count == 1 ? "a column" : "columns")} that {Path.GetFileName(reportedAs ?? dataFile)} does not have: {named}. Available columns: {string.Join(", ", columns)}.";
        var sameLetters = absent.Select(name => columns.FirstOrDefault(column => string.Equals(column, name, StringComparison.OrdinalIgnoreCase)))
            .FirstOrDefault(column => column is not null);
        return sameLetters is null ? message : $"{message} '{sameLetters}' differs only in case; column names are case-sensitive.";
    }

    /// <summary>
    /// What a header that came out as one column is separated by instead - a semicolon (the spreadsheet export of a locale that writes
    /// decimals with a comma), a tab or a bar - or <c>null</c> when it is not that: more than one column, or none of them in the name.
    /// </summary>
    internal static string? OtherDelimiter(IReadOnlyList<string> columns)
    {
        if (columns.Count != 1)
            return null;
        var name = columns[0];
        var best = new[] { (Name: "a semicolon (;)", Char: ';'), (Name: "a tab", Char: '\t'), (Name: "a bar (|)", Char: '|') }
            .Select(d => (d.Name, Count: name.Count(c => c == d.Char)))
            .OrderByDescending(d => d.Count)
            .First();
        return best.Count >= 1 ? best.Name : null;
    }

    /// <summary>
    /// Why a data file with no content cannot be trained on, or <c>null</c>. Said first and as that: an empty table has no header to
    /// name a label by, and the first reader to meet it (the parser behind the format conversion) reported its own state dump -
    /// "No header record was found" with the reader's column count and byte offsets - which names neither the file nor the cause.
    /// A file with a header and no rows is <see cref="HeaderOnlyDataFileRefusal"/>'s. Directory datasets have no
    /// single file to be empty.
    /// </summary>
    public static string? EmptyDataFileRefusal(string dataFile, string? task, string? reportedAs = null)
    {
        if (DataLoaderFactory.IsDirectoryBased(task) || !File.Exists(dataFile))
            return null;
        try
        {
            var name = Path.GetFileName(reportedAs ?? dataFile);
            if (new FileInfo(dataFile).Length == 0)
                return $"{name} is empty: it has no header row and no data.";
            var text = dataFile.EndsWith(".csv", StringComparison.OrdinalIgnoreCase) || dataFile.EndsWith(".tsv", StringComparison.OrdinalIgnoreCase)
                || dataFile.EndsWith(".txt", StringComparison.OrdinalIgnoreCase);
            if (!text)
                return null;
            using var reader = EncodingDetector.OpenReader(dataFile);
            var buffer = new char[4096];
            int read;
            while ((read = reader.Read(buffer, 0, buffer.Length)) > 0)
            {
                for (var i = 0; i < read; i++)
                {
                    if (!char.IsWhiteSpace(buffer[i]) && buffer[i] != '\uFEFF')
                        return null;
                }
            }
            return $"{name} is empty: it holds only blank space, no header row and no data.";
        }
        catch (IOException) { return null; }
        catch (UnauthorizedAccessException) { return null; }
    }

    /// <summary>
    /// Why a table file that holds its header row and nothing under it cannot be trained on, or <c>null</c>. Said as that: left
    /// to the steps that count rows, the file trained as a label with 0 values and ended in "Cannot train a classifier with only one
    /// class" - advice about the label for a file that has no rows. Only comma- and tab-separated files are read here (a record is
    /// counted as CSV, so a quoted header that spans lines is one), and a file with no header at all is
    /// <see cref="EmptyDataFileRefusal"/>'s. A workbook or Parquet file is read once converted: the converted copy is passed as
    /// <paramref name="dataFile"/> and the file the user named as <paramref name="reportedAs"/>, so the refusal names the user's file
    /// (a header-only workbook otherwise trained as "only one class" - the advice about the label this refusal exists to prevent). A
    /// converted table with no header at all never gets here: <see cref="TabularDataFile.AsCsvAsync"/> refuses it as empty.
    /// </summary>
    public static string? HeaderOnlyDataFileRefusal(string dataFile, string? task, string? reportedAs = null)
    {
        if (DataLoaderFactory.IsDirectoryBased(task) || !File.Exists(dataFile))
            return null;
        if (!dataFile.EndsWith(".csv", StringComparison.OrdinalIgnoreCase) && !dataFile.EndsWith(".tsv", StringComparison.OrdinalIgnoreCase))
            return null;
        try
        {
            // The first record is the header; a record under it is a row unless it is a blank line (TableRows.IsRow: in a table of
            // several columns a line of white space is none, the parser reads it as one empty field).
            string[]? header = null;
            foreach (var record in MLoop.Core.Data.TableRows.ReadRecords(dataFile, includeHeader: true))
            {
                if (header is null)
                {
                    if (record.All(string.IsNullOrWhiteSpace))
                        return null; // no header row at all: EmptyDataFileRefusal's
                    header = record;
                    continue;
                }
                if (header.Length == 1 || record.Length != 1 || !string.IsNullOrWhiteSpace(record[0]))
                    return null;
            }
            return header is null ? null : $"{Path.GetFileName(reportedAs ?? dataFile)} has a header row and no data rows.";
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or CsvHelper.CsvFormatException or CsvHelper.CsvValidationException)
        {
            // A file that cannot be read as a table is reported by the step that reads it.
            return null;
        }
    }

    /// <summary>What to do when <see cref="HeaderOnlyDataFileRefusal"/> is the answer.</summary>
    public const string HeaderOnlyDataFileGuidance =
        "Put the data rows under the header row - or train on another file: mloop train <data-file>";

    /// <summary>What to do when <see cref="EmptyDataFileRefusal"/> is the answer.</summary>
    public const string EmptyDataFileGuidance =
        "Put a header row naming the columns, and the data rows under it, in the file - or train on another file: mloop train <data-file>";

    /// <summary>
    /// Why a table cannot be trained on because two of its columns share a name, or <c>null</c>. A column is found by its name - by the
    /// loader, the saved schema, a prediction's input - so the second of two columns named alike either shadows the first or fails
    /// after the whole search ("An item with the same key has already been added"), and the model it would have made is lost. Empty
    /// names are not counted: an index column has none and is removed. A name written over two lines is read with the break as a space,
    /// and names are compared as the same text however composed, ignoring nothing else - column names are case-sensitive.
    /// </summary>
    public static string? DuplicateColumnRefusal(string dataFile, string? task, string? reportedAs = null)
    {
        if (DataLoaderFactory.IsDirectoryBased(task))
            return null;
        var header = HeaderRecord(dataFile);
        if (header is null)
            return null;

        var named = header
            .Select((name, index) => (Name: MLoop.Core.Data.CanonicalText.Nfc(name), Position: index + 1))
            .Where(c => c.Name.Length > 0)
            .GroupBy(c => c.Name, StringComparer.Ordinal)
            .Where(g => g.Count() > 1)
            .ToList();
        if (named.Count == 0)
            return null;
        var said = string.Join("; ", named.Take(3).Select(g => $"'{g.Key}' is columns {string.Join(", ", g.Select(c => c.Position))}"));
        return $"{Path.GetFileName(reportedAs ?? dataFile)} has columns that share a name: {said}{(named.Count > 3 ? "; …" : "")}. A column is found by its name, so each name must be its own.";
    }

    /// <summary>
    /// The names in the first record of a CSV or TSV file, read in the file's own encoding, a name broken over two lines as one name with
    /// a space; <c>null</c> for another format, a missing or an empty file.
    /// </summary>
    internal static string[]? HeaderRecord(string dataFile)
    {
        var delimiter = dataFile.EndsWith(".tsv", StringComparison.OrdinalIgnoreCase) ? "\t"
            : dataFile.EndsWith(".csv", StringComparison.OrdinalIgnoreCase) ? ","
            : null;
        if (delimiter is null)
            return null;
        try
        {
            using var reader = EncodingDetector.OpenReader(dataFile);
            using var csv = CsvHelper.CsvReader.Create(reader, MLoop.Core.Data.CsvDialect.Reader(hasHeader: false, separator: delimiter[0]));
            return csv.Read()
                ? [.. MLoop.Core.Data.CsvDialect.Record(csv).Select(name => System.Text.RegularExpressions.Regex.Replace(name.TrimStart('\uFEFF'), @"\r\n|\r|\n", " "))]
                : null;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or FormatException)
        {
            return null;
        }
    }

    /// <summary>What to do when <see cref="LabelColumnRefusal"/> is the answer.</summary>
    public const string LabelColumnGuidance =
        "Name a column of the file exactly as its header spells it: mloop train <data-file> --label <column> (or set the label in mloop.yaml).";

    /// <summary>What to do when <see cref="DuplicateColumnRefusal"/> is the answer.</summary>
    public const string DuplicateColumnGuidance = "Rename one of them in the file, or delete the column you do not need.";

    /// <summary>A refusal and, when there is one, the way out.</summary>
    public sealed record Refusal(string Message, string? Hint = null);

    /// <summary>
    /// What a model definition lacks or gets wrong before any data is read: the task absent or not one the
    /// tool knows, a label a supervised task needs, the columns ranking, recommendation and forecasting
    /// need. One answer for <c>mloop train</c> and <c>POST /train</c> — the server used to check none of it,
    /// so a misspelled task was answered 202 and trained as whichever task a predicate happened to match.
    /// </summary>
    public static Refusal? DefinitionRefusal(ModelDefinition definition)
    {
        ArgumentNullException.ThrowIfNull(definition);
        // TrainingRunner.RequiresLabel is the single source for which tasks need a label.
        if (string.IsNullOrEmpty(definition.Task) ||
            (TrainingRunner.LabelColumnMustBeNamed(definition.Task) && string.IsNullOrEmpty(definition.Label)))
            return new Refusal("Task is required. Label is required for supervised tasks. Use --task and <label> arguments, or define them in mloop.yaml");

        // The spelling is already canonical by now (ConfigMerger), so a name that fails here is
        // one no spelling of any known task produces.
        if (!TaskTypes.IsValid(definition.Task))
            return new Refusal($"Invalid task type '{definition.Task}'.", $"Valid options: {TaskTypes.Listed}");

        if (definition.Task.Equals("ranking", StringComparison.OrdinalIgnoreCase) && string.IsNullOrEmpty(definition.GroupColumn))
            return new Refusal("Ranking task requires a group column. Use --group-column or set 'group_column' in mloop.yaml");

        if (definition.Task.Equals("recommendation", StringComparison.OrdinalIgnoreCase) &&
            (string.IsNullOrEmpty(definition.UserColumn) || string.IsNullOrEmpty(definition.ItemColumn)))
            return new Refusal("Recommendation task requires user_column and item_column. Set them in mloop.yaml");

        if (definition.Task.Equals("forecasting", StringComparison.OrdinalIgnoreCase) && (definition.Horizon ?? 0) <= 0)
            return new Refusal("Forecasting task requires horizon > 0. Set 'horizon' in mloop.yaml");

        return null;
    }

    /// <summary>
    /// Why options that shape a table cannot be given to a task that trains on a directory of images — or <c>null</c>. Each would
    /// do nothing there (no rows to sample, merge, drop, analyze or seed), and an option that silently did nothing would leave its
    /// user believing the run was shaped by it. <paramref name="given"/> names the options as the caller spells them.
    /// </summary>
    public static string? TableOnlyOptionsRefusal(string? task, IEnumerable<string> given)
    {
        if (!DataLoaderFactory.IsDirectoryBased(task))
            return null;
        var names = given.ToList();
        return names.Count == 0
            ? null
            : $"{string.Join(", ", names)} shape{(names.Count == 1 ? "s" : "")} a table's rows; '{task}' trains on a directory of images, where {(names.Count == 1 ? "it" : "they")} would do nothing.";
    }

    /// <summary>
    /// The directory a directory-based task trains on (image classification: class subfolders; object
    /// detection: a COCO annotations file plus its images): the path given, resolved against the project, else
    /// the task's convention — <c>datasets/coco</c> for object detection, <c>datasets/images</c> otherwise,
    /// both finally <c>datasets</c>. <c>null</c> when there is none. One answer for <c>mloop train</c> and
    /// <c>POST /train</c>.
    /// </summary>
    public static string? ResolveDirectoryDataset(string task, string? dataFile, string[]? dataPaths, string projectRoot)
    {
        if (!string.IsNullOrWhiteSpace(dataFile))
        {
            var p = ProjectPaths.Resolve(dataFile, projectRoot);
            return Directory.Exists(p) ? p : null;
        }

        if (dataPaths is { Length: > 0 } && !string.IsNullOrWhiteSpace(dataPaths[0]))
        {
            var p = ProjectPaths.Resolve(dataPaths[0], projectRoot);
            return Directory.Exists(p) ? p : null;
        }

        var conventionDir = task.Equals("object-detection", StringComparison.OrdinalIgnoreCase)
            ? "coco"
            : "images";

        foreach (var candidate in new[]
        {
            Path.Combine(projectRoot, ProjectLayout.DatasetsDirectory, conventionDir),
            Path.Combine(projectRoot, ProjectLayout.DatasetsDirectory)
        })
        {
            if (Directory.Exists(candidate))
                return candidate;
        }

        return null;
    }

    /// <summary>Why no dataset directory was found for a directory-based task, and where to put one.</summary>
    public static Refusal DirectoryDatasetNotFound(string task) =>
        task.Equals("object-detection", StringComparison.OrdinalIgnoreCase)
            ? new Refusal("Object-detection dataset not found.",
                "Put a COCO annotations.json plus images under datasets/coco/, or pass a directory: mloop train --task object-detection <dir>")
            : new Refusal("Image dataset directory not found.",
                "Lay images out as datasets/images/<class>/<files>, or pass a directory: mloop train --task image-classification <dir>");

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
        // A workbook or Parquet file with a header and no rows is only a table now: said as that, as for a CSV of the same shape.
        if (HeaderOnlyDataFileRefusal(file, definition.Task, reportedAs: dataFile) is { } headerOnlyRefusal)
            throw new InvalidOperationException($"{headerOnlyRefusal} {HeaderOnlyDataFileGuidance}");
        file = CsvDataLoader.FlattenMultiLineQuotedFields(file, reportedAs: dataFile);

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
        "Check that --label (or the label in mloop.yaml) names the column you meant, or train on a file that holds more than one class.";

    /// <summary>
    /// The refusal when a binary-classification task is given a label of more than two values, or null. Every trainer of the
    /// search fails on such a label ("Schema mismatch for label column: expected Boolean"), and the run said so only after the
    /// whole first round of trials had failed. Values are compared the way the binary label is read: ignoring case, and as
    /// the same text however it is composed.
    /// </summary>
    public static string? BinaryLabelRefusal(string? task, string label, IEnumerable<string> classNames)
    {
        ArgumentNullException.ThrowIfNull(classNames);
        if (!string.Equals(MLoop.Core.Models.TaskTypes.Canonical(task), "binary-classification", StringComparison.Ordinal))
            return null;
        var values = classNames.Select(MLoop.Core.Data.CanonicalText.Nfc).Distinct(StringComparer.OrdinalIgnoreCase).ToList();
        if (values.Count <= 2)
            return null;
        var examples = string.Join(", ", values.Take(3).Select(v => $"'{v}'"));
        return $"A binary-classification model chooses between two classes, but the label '{label}' holds {values.Count} different values ({examples}{(values.Count > 3 ? ", …" : "")}).";
    }

    /// <summary>What to do when <see cref="BinaryLabelRefusal"/> is the answer.</summary>
    public const string BinaryLabelGuidance =
        "Train it as multiclass-classification if each value is a class, as regression if the label is a number to estimate " +
        "(the task is set by `mloop init --task` or `task:` in mloop.yaml), or reduce the label to two values.";

    /// <summary>
    /// Why <paramref name="balance"/> cannot be honoured for <paramref name="definition"/>, or null: not a balance
    /// option, a task that is not classification, or a weight column that already decides how often rows count. An
    /// option the run silently ignored would train a model its user believes balanced.
    /// </summary>
    public static string? BalanceConflict(ModelDefinition definition, string? balance)
    {
        ArgumentNullException.ThrowIfNull(definition);
        if (ClassBalance.OptionError(balance) is { } optionError)
            return optionError;
        if (string.IsNullOrEmpty(balance) || balance.Equals("none", StringComparison.OrdinalIgnoreCase))
            return null;
        if (!TrainingRunner.IsClassification(definition.Task))
            return DataLoaderFactory.IsDirectoryBased(definition.Task)
                ? $"--balance oversamples whole examples by their one class; '{definition.Task}' labels the boxes in an image, so an image has no single class to balance."
                : $"--balance applies to classification tasks; '{definition.Task}' has no classes to balance.";
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
