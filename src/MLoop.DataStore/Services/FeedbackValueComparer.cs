using System.Text.Json;

namespace MLoop.DataStore.Services;

/// <summary>
/// Single source of truth for deciding whether a feedback entry's predicted value matches its
/// actual (ground-truth) value. Both the accuracy calculation (<see cref="FileFeedbackCollector"/>)
/// and the <c>mloop feedback list</c> "Match" column must use the same rule — previously each had
/// its own copy and they drifted: the collector's handled the runtime reality that feedback values
/// round-trip through JSON as <see cref="JsonElement"/>, while the CLI's naive copy compared the raw
/// <c>JsonElement</c> structs (never equal across documents), so the list always showed ✗ even when
/// the metrics command reported 100% accuracy.
/// </summary>
public static class FeedbackValueComparer
{
    /// <summary>
    /// True when <paramref name="predicted"/> and <paramref name="actual"/> represent the same value.
    /// Values stored as <see cref="JsonElement"/> are normalized first; strings compare
    /// case-insensitively (labels); numbers compare with a small tolerance.
    /// </summary>
    public static bool ValuesMatch(object? predicted, object? actual)
    {
        if (predicted == null && actual == null)
            return true;
        if (predicted == null || actual == null)
            return false;

        if (predicted is JsonElement predictedJson)
            predicted = Normalize(predictedJson);
        if (actual is JsonElement actualJson)
            actual = Normalize(actualJson);

        // Two text labels: case-insensitive (e.g. "OK" / "ok").
        if (predicted is string predictedStr && actual is string actualStr)
            return string.Equals(predictedStr, actualStr, StringComparison.OrdinalIgnoreCase);

        // Numeric comparison whenever both sides represent a number — including the common cross-type
        // case where the prediction was logged as a JSON number but the CLI-supplied actual is its
        // string form (e.g. predicted 1 vs actual "1", or 3.14 vs "3.14"). Without this, accuracy was
        // always 0 for regression and numeric-label models.
        if (TryAsDouble(predicted, out var predictedNum) && TryAsDouble(actual, out var actualNum))
            return Math.Abs(predictedNum - actualNum) < 0.0001;

        // Fall back to a culture-invariant textual comparison of the two representations.
        return string.Equals(
            predicted.ToString(), actual.ToString(), StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Whether <paramref name="actual"/> lies inside <paramref name="interval"/> — the judgement a prediction that stated an interval
    /// is read by, in place of <see cref="ValuesMatch"/> (two numbers are never equal). <see langword="null"/> when it cannot be
    /// judged: no interval, one that cannot hold a value (<see cref="Interfaces.PredictionInterval.CanJudge"/>), or an actual value that is not
    /// a number. The one rule — the metrics and the <c>feedback list</c> Match column both ask it.
    /// </summary>
    public static bool? InsideInterval(Interfaces.PredictionInterval? interval, object? actual) =>
        interval is { } stated && stated.CanJudge() && TryReadNumber(actual, out var number) ? stated.Contains(number) : null;

    /// <summary>
    /// Reads <paramref name="value"/> as a finite number — a number, a JSON number or its text — the way
    /// <see cref="ValuesMatch"/> reads one, so an actual value is judged by the same reading everywhere.
    /// </summary>
    public static bool TryReadNumber(object? value, out double number)
    {
        number = 0;
        if (value is null)
            return false;
        if (value is JsonElement element)
            value = Normalize(element);
        return TryAsDouble(value, out number) && double.IsFinite(number);
    }

    private static bool TryAsDouble(object value, out double result)
    {
        switch (value)
        {
            case byte or sbyte or short or ushort or int or uint or long or ulong or float or double or decimal:
                result = Convert.ToDouble(value, System.Globalization.CultureInfo.InvariantCulture);
                return true;
            case string s:
                return double.TryParse(
                    s, System.Globalization.NumberStyles.Any,
                    System.Globalization.CultureInfo.InvariantCulture, out result);
            default:
                result = 0;
                return false;
        }
    }

    private static object Normalize(JsonElement element)
    {
        return element.ValueKind switch
        {
            JsonValueKind.String => element.GetString() ?? string.Empty,
            JsonValueKind.Number => element.TryGetInt64(out var l) ? l : element.GetDouble(),
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            JsonValueKind.Null => new object(),
            _ => element.GetRawText()
        };
    }
}
