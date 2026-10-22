namespace MLoop.Core.Data;

/// <summary>
/// Whether a cell's text reads as a float or a boolean, by the rules ML.NET's own conversions use when
/// it loads a file — which <see cref="ColumnLayoutInference"/> must follow to type a column the way the
/// loader will read it. These are not <see cref="float.TryParse(string, out float)"/>: no culture,
/// <c>?</c> and <c>NaN</c> are numbers, <c>NA</c> and <c>N/A</c> are missing numbers, <c>+Infinity</c>
/// is not one, and <c>yes</c> or a lone <c>+</c> is a boolean.
/// </summary>
internal static class ColumnLayoutValues
{
    private const char InfinitySymbol = '∞';

    /// <summary>Whether <paramref name="span"/> reads as a float (an empty value reads as one).</summary>
    public static bool IsSingle(ReadOnlySpan<char> span) =>
        ParsesAsNumber(span) || (!span.IsEmpty && IsStandardMissing(span));

    private static bool ParsesAsNumber(ReadOnlySpan<char> span)
    {
        var ich = 0;
        for (; ; ich++)
        {
            if (ich >= span.Length)
                return true;
            if (!char.IsWhiteSpace(span[ich]))
                break;
        }

        if (span.Length - ich == 1 && span[ich] is >= '0' and <= '9' or '?')
            return true;

        var number = span[ich..];
        var end = 0;
        if (!ParseNumber(number, ref end))
        {
            end = 0;
            if (!ParseSpecial(number, ref end))
                return false;
        }

        // The end is counted within the number but checked against the whole value, as ML.NET does:
        // after leading white space, a number longer than one character is refused unless the
        // characters that many places from the start are white space too. Only a quoted value reaches
        // here with leading white space — the loader drops it before an unquoted field.
        for (; end < span.Length; end++)
            if (!char.IsWhiteSpace(span[end]))
                return false;
        return true;
    }

    // Sign, digits, an optional decimal point and digits, an optional exponent. On success the end is
    // where the number stops; what follows is the caller's to judge.
    private static bool ParseNumber(ReadOnlySpan<char> span, ref int end)
    {
        if (span.IsEmpty)
            return false;

        var digits = false;
        var i = 0;
        switch (span[i])
        {
            case '-':
            case '+':
                if (++i >= span.Length)
                    return false;
                break;
            case '.':
                goto Point;
            case >= '0' and <= '9':
                break;
            default:
                return false;
        }

        for (; ; )
        {
            if (!IsDigit(span[i]))
                break;
            digits = true;
            if (++i >= span.Length)
            {
                end = i;
                return true;
            }
        }

        if (span[i] != '.')
            goto AfterDigits;

    Point:
        for (; ; )
        {
            if (++i >= span.Length)
            {
                if (digits)
                    end = i;
                return digits;
            }
            if (!IsDigit(span[i]))
                break;
            digits = true;
        }

    AfterDigits:
        if (!digits)
            return false;

        end = i;
        if (span[i] is not ('e' or 'E'))
            return true;
        if (++i >= span.Length)
            return true;
        if (span[i] is '-' or '+' && ++i >= span.Length)
            return true;

        var exponentDigits = false;
        for (; ; )
        {
            if (!IsDigit(span[i]))
                break;
            exponentDigits = true;
            if (++i >= span.Length)
                break;
        }
        if (exponentDigits)
            end = i;
        return true;
    }

    private static bool ParseSpecial(ReadOnlySpan<char> span, ref int end)
    {
        if (span.IsEmpty)
            return false;

        switch (span[0])
        {
            case '?':
            case InfinitySymbol:
                end = 1;
                return true;
            case 'N' when span.StartsWith("NaN"):
                end = 3;
                return true;
            case 'I' when span.StartsWith("Infinity"):
                end = 8;
                return true;
            case '-' when span.Length >= 2 && span[1] == InfinitySymbol:
                end = 2;
                return true;
            case '-' when span.StartsWith("-Infinity"):
                end = 9;
                return true;
            default:
                return false;
        }
    }

    // ?, NA, N/A and NaN in any case: a missing number.
    private static bool IsStandardMissing(ReadOnlySpan<char> span) => span.Length switch
    {
        1 => span[0] == '?',
        2 => span.Equals("NA", StringComparison.OrdinalIgnoreCase),
        3 => span.Equals("N/A", StringComparison.OrdinalIgnoreCase) || span.Equals("NaN", StringComparison.OrdinalIgnoreCase),
        _ => false,
    };

    /// <summary>
    /// Whether <paramref name="span"/> reads as a boolean: empty, one of <c>T t Y y 1 +</c> /
    /// <c>F f N n 0 -</c>, <c>+1</c> / <c>-1</c>, or <c>yes</c> / <c>no</c> / <c>true</c> / <c>false</c>
    /// in any case.
    /// </summary>
    public static bool IsBoolean(ReadOnlySpan<char> span) => span.Length switch
    {
        0 => true,
        1 => span[0] is 'T' or 't' or 'Y' or 'y' or '1' or '+' or 'F' or 'f' or 'N' or 'n' or '0' or '-',
        2 => span.Equals("no", StringComparison.OrdinalIgnoreCase) || span is "+1" or "-1",
        3 => span.Equals("yes", StringComparison.OrdinalIgnoreCase),
        4 => span.Equals("true", StringComparison.OrdinalIgnoreCase),
        5 => span.Equals("false", StringComparison.OrdinalIgnoreCase),
        _ => false,
    };

    private static bool IsDigit(char ch) => ch is >= '0' and <= '9';
}
