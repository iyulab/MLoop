using System.Globalization;
using System.Text.RegularExpressions;

namespace MLoop.Core.Data;

/// <summary>
/// Whether a column of text is really numbers written for people: <c>1,234.5</c>, <c>12%</c>, <c>$3</c>, <c>₩5,000</c>.
/// A column like that is not a number to the loader, so it is read as words, and a model that needed its magnitude
/// has none (a regression on a price written with its thousands separator scored R² 0.9997 as <c>1234.5</c> and −0.004
/// as <c>"1,234.5"</c>, with nothing in the run saying why). The question is asked of the values of a column the
/// loader read as text, in one place, so a report and a warning cannot disagree about which columns these are.
/// </summary>
public static partial class FormattedNumbers
{
    /// <summary>Fewest non-empty values a column needs before it is judged — a handful of values is not a column's habit.</summary>
    public const int MinimumValues = 5;

    /// <summary>The share of non-empty values that must read as numbers in a form of this kind.</summary>
    public const double MinimumShare = 0.9;

    // An optional sign, an optional currency symbol, digits (with thousands separators or not), a decimal part, an optional percent
    // sign or currency word. A lone plain number matches too: whether the column is formatted is asked of its values together.
    [GeneratedRegex(@"^[+-]?\s*[$€£¥₩]?\s*(\d{1,3}(,\d{3})+|\d+)(\.\d+)?\s*(%|원)?$", RegexOptions.CultureInvariant)]
    private static partial Regex Form();

    /// <summary>Whether one value is a number in one of the written forms, plain or formatted.</summary>
    public static bool IsNumberInAnyForm(string value) => Form().IsMatch(value.Trim());

    /// <summary>Whether one value is such a number and carries something the loader does not read as one (a separator, <c>%</c>, a currency sign).</summary>
    public static bool IsFormatted(string value)
    {
        var trimmed = value.Trim();
        return Form().IsMatch(trimmed) && !double.TryParse(trimmed, NumberStyles.Float, CultureInfo.InvariantCulture, out _);
    }

    /// <summary>
    /// Whether the values of a text column are numbers written with a separator, <c>%</c> or a currency sign: at
    /// least <see cref="MinimumValues"/> non-empty values, <see cref="MinimumShare"/> of them numbers in a written form,
    /// and at least one of those formatted (a column of plain numbers is numeric already and is not asked about).
    /// </summary>
    public static bool Looks(IEnumerable<string> values, out string? example)
    {
        ArgumentNullException.ThrowIfNull(values);
        var nonEmpty = 0;
        var numbers = 0;
        example = null;
        foreach (var value in values)
        {
            if (string.IsNullOrWhiteSpace(value))
                continue;
            nonEmpty++;
            if (!IsNumberInAnyForm(value))
                continue;
            numbers++;
            if (example is null && IsFormatted(value))
                example = value.Trim();
        }

        var judged = nonEmpty >= MinimumValues && numbers >= MinimumShare * nonEmpty && example is not null;
        if (!judged)
            example = null;
        return judged;
    }
}
