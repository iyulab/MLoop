using System.Globalization;
using System.Text.RegularExpressions;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Data;
using MLoop.Core.Models;

namespace MLoop.Core.Training;

/// <summary>
/// A label that has to be a number but that the loader read as text — what that means, said before the trainers
/// find it out. A regression or a recommendation fits a number; when the loader types the label column as text (one value
/// such as <c>1,234.5</c> among the first rows is enough), every trainer fails with <c>Schema mismatch for label column
/// 'y': expected Single, got String</c>, four times over, and the run ends without a word about which values did it. A ranking
/// converts its relevance grades to numbers and turned each such value into a missing one, with a warning and nothing else.
/// </summary>
/// <remarks>
/// The question is asked of the loaded data, not of the file: what the loader typed the column as is what the trainers
/// get, so there is no second list of tokens to keep in step with the loader's. <c>N/A</c>, <c>NA</c>, <c>NaN</c> and
/// <c>?</c> are missing numbers to the loader and never make the label text; a number written for people
/// (<see cref="FormattedNumbers"/>) does, and rows are not quietly dropped for it — the large values are the ones that
/// carry the separator, so the rows that would be lost are the biased ones.
/// </remarks>
public static partial class NumericLabelCheck
{
    // 1,5 or -0,25: digits, a comma, one or two digits — not a thousands separator (three digits), which FormattedNumbers names.
    [GeneratedRegex(@"^[+-]?\d+,\d{1,2}$", RegexOptions.CultureInvariant)]
    private static partial Regex DecimalComma();

    /// <summary>How many rows are read to find and count the values the loader could not read.</summary>
    public const int SampleRows = 100_000;

    /// <summary>
    /// Whether the label of <paramref name="task"/> is a number: it reads a table, needs a label, and that label is neither a class nor
    /// text — regression, forecasting, ranking, recommendation, sentence similarity — of a task this tool knows (a name nobody knows
    /// has no label to read). The same distinction the data check makes.
    /// </summary>
    public static bool HasNumericLabel(string task)
        => TaskTypes.IsValid(task) && TrainingRunner.LabelColumnMustBeNamed(task) && !TrainingRunner.IsClassification(task) && !TrainingRunner.HasTextLabel(task);

    /// <summary>
    /// The refusal for <paramref name="labelColumn"/> of <paramref name="data"/>, or <see langword="null"/> when the task
    /// does not need a numeric label or the label is numeric.
    /// </summary>
    public static string? Refusal(IDataView data, string task, string labelColumn)
    {
        ArgumentNullException.ThrowIfNull(data);
        if (!HasNumericLabel(task))
            return null;
        if (data.Schema.GetColumnOrNull(labelColumn) is not { Type: TextDataViewType } column)
            return null;

        var unreadable = 0;
        var seen = 0;
        string? example = null;
        foreach (var memory in data.GetColumn<ReadOnlyMemory<char>>(column).Take(SampleRows))
        {
            var value = memory.Span.Trim();
            if (value.IsEmpty)
                continue;
            seen++;
            if (ReadsAsNumberOrMissing(value))
                continue;
            unreadable++;
            example ??= value.ToString();
        }

        var counted = seen == SampleRows
            ? $"at least {unreadable:N0} of the first {SampleRows:N0} label values in the training rows"
            : $"{unreadable:N0} of the {seen:N0} label values in the training rows";
        return Message(labelColumn, task, counted, example);
    }

    /// <summary>
    /// The refusal for a label column whose values the loader cannot read as numbers: <paramref name="counted"/> says how many
    /// and over what, <paramref name="example"/> is the first such value. One wording for the check on the loaded data (above)
    /// and the one over every row of the file (<c>DataQualityValidator</c>) — the loader types a column from the start of
    /// the file, so a value further down loads as a missing number and only the file shows it.
    /// </summary>
    public static string Message(string labelColumn, string task, string counted, string? example, bool columnReadAsText = true)
    {
        // The column is text when the start of the file already holds such values; further down it is numbers with holes.
        var lead = columnReadAsText
            ? $"Label column '{labelColumn}' is read as text, not as numbers: {counted}"
            : $"Label column '{labelColumn}' holds values that are not read as numbers: {counted}";
        if (example is not null && FormattedNumbers.IsFormatted(example))
            return $"{lead} are numbers written for people " +
                   $"(for example '{example}' — a thousands separator, a percent or currency sign, or full-width digits). The loader does not read " +
                   (columnReadAsText
                       ? $"those as numbers, and a {task} model needs the label's size, so no trainer can fit it; dropping the rows would leave out the large values. "
                       : $"those as numbers: each loads as a missing label and its row is dropped, and the large values are the ones written this way, so the {task} model would never see them. ") +
                   "Write the column as plain numbers (1234.5 for 1,234.5) and train again.";
        if (example is not null && DecimalComma().IsMatch(example))
            return $"{lead} are written with a decimal comma " +
                   $"(for example '{example}'). The loader reads a point as the decimal mark and anything else as text, and a {task} model needs the label's size. " +
                   "Write them with a point (1.5 for 1,5) and train again.";
        return $"{lead} are not numbers" +
               (example is null ? "." : $" (for example '{example}').") +
               $" A {task} label has to be numeric. If it names classes, train with --task multiclass-classification or binary-classification instead; " +
               "otherwise fix or empty those values (an empty value, N/A, NA, NaN or ? is read as a missing number).";
    }

    /// <summary>
    /// The label cells of a file, counted by what the loader makes of them: a number, a missing number, or a value it cannot
    /// read. One count for the check before training and the one before a file is scored, so they refuse the same values.
    /// </summary>
    public sealed class Census
    {
        /// <summary>Label cells that are not empty.</summary>
        public long Labelled { get; private set; }

        /// <summary>Label cells that read as a number.</summary>
        public long Numbers { get; private set; }

        /// <summary>Label cells that are neither a number nor a missing number (N/A, NA, NaN, ?).</summary>
        public long Unreadable { get; private set; }

        /// <summary>Label cells written as a missing number (N/A, NA, NaN, ?) — read as missing, so the row has no label.</summary>
        public long MissingTokens { get; private set; }

        /// <summary>The first such cell, as written.</summary>
        public string? Example { get; private set; }

        /// <summary>Counts one cell.</summary>
        public void Add(ReadOnlySpan<char> cell)
        {
            var value = cell.Trim();
            if (value.IsEmpty)
                return;
            Labelled++;
            if (double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out _))
            {
                Numbers++;
                return;
            }
            if (ReadsAsNumberOrMissing(value))
            {
                MissingTokens++;
                return;
            }
            Unreadable++;
            Example ??= value.ToString();
        }

        /// <summary>The refusal for what was counted, or <see langword="null"/> when every value reads.</summary>
        public string? Refusal(string labelColumn, string task) => Unreadable == 0
            ? null
            : Message(labelColumn, task, $"{Unreadable:N0} of the {Labelled:N0} label values in the file", Example, columnReadAsText: false);
    }

    /// <summary>
    /// The refusal for the label of the table in <paramref name="csvPath"/> (UTF-8, one record per line) read before the file is
    /// scored — <c>evaluate</c>, <c>explain</c>, <c>threshold</c>, <c>calibrate</c> — or <see langword="null"/>: a task whose label is
    /// not a number, no such column in the file, or every value reads. A value the loader cannot read loads as a missing number,
    /// so the metrics would be computed on the rows it left, or on a label it made up, with nothing said.
    /// </summary>
    public static string? FileRefusal(string csvPath, string labelColumn, string task)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(csvPath);
        if (string.IsNullOrEmpty(labelColumn) || !HasNumericLabel(task))
            return null;

        using var reader = new StreamReader(csvPath, System.Text.Encoding.UTF8, detectEncodingFromByteOrderMarks: true);
        if (reader.ReadLine() is not { } header)
            return null;
        var names = Core.Prediction.CsvFieldParser.ParseFields(header);
        var index = Array.FindIndex(names, n => string.Equals(n.Trim(), labelColumn, StringComparison.OrdinalIgnoreCase));
        if (index < 0)
            return null;

        var census = new Census();
        while (reader.ReadLine() is { } line)
        {
            if (!TableRows.IsRow(line, names.Length))
                continue;
            var fields = Core.Prediction.CsvFieldParser.ParseFields(line);
            if (index < fields.Length)
                census.Add(fields[index]);
        }
        return census.Refusal(labelColumn, task);
    }

    /// <summary>The loader's reading of a cell as a float, or as a missing number (<c>ColumnLayoutValues</c>: ?, NA, N/A, NaN in any case).</summary>
    public static bool ReadsAsNumberOrMissing(ReadOnlySpan<char> value) =>
        double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out _)
        || value is "?"
        || value.Equals("NA", StringComparison.OrdinalIgnoreCase)
        || value.Equals("N/A", StringComparison.OrdinalIgnoreCase)
        || value.Equals("NaN", StringComparison.OrdinalIgnoreCase);
}
