using System.Globalization;
using System.Text.RegularExpressions;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLoop.Core.Data;

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
    /// text — regression, forecasting, ranking, recommendation, sentence similarity. The same distinction the data check makes.
    /// </summary>
    public static bool HasNumericLabel(string task)
        => TrainingRunner.LabelColumnMustBeNamed(task) && !TrainingRunner.IsClassification(task) && !TrainingRunner.HasTextLabel(task);

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
        if (example is not null && FormattedNumbers.IsFormatted(example))
            return $"Label column '{labelColumn}' is read as text, not as numbers: {counted} are numbers written for people " +
                   $"(for example '{example}' — a thousands separator, a percent or currency sign, or full-width digits). The loader does not read " +
                   $"those as numbers, and a {task} model needs the label's size, so no trainer can fit it; dropping the rows would leave out the large values. " +
                   "Write the column as plain numbers (1234.5 for 1,234.5) and train again.";
        if (example is not null && DecimalComma().IsMatch(example))
            return $"Label column '{labelColumn}' is read as text, not as numbers: {counted} are written with a decimal comma " +
                   $"(for example '{example}'). The loader reads a point as the decimal mark and anything else as text, and a {task} model needs the label's size. " +
                   "Write them with a point (1.5 for 1,5) and train again.";
        return $"Label column '{labelColumn}' is read as text, not as numbers: {counted} are not numbers" +
               (example is null ? "." : $" (for example '{example}').") +
               $" A {task} label has to be numeric. If it names classes, train with --task multiclass-classification or binary-classification instead; " +
               "otherwise fix or empty those values (an empty value, N/A, NA, NaN or ? is read as a missing number).";
    }

    // The loader's reading of a cell as a float, or as a missing number (ColumnLayoutValues: ?, NA, N/A, NaN in any case).
    private static bool ReadsAsNumberOrMissing(ReadOnlySpan<char> value) =>
        double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out _)
        || value is "?"
        || value.Equals("NA", StringComparison.OrdinalIgnoreCase)
        || value.Equals("N/A", StringComparison.OrdinalIgnoreCase)
        || value.Equals("NaN", StringComparison.OrdinalIgnoreCase);
}
