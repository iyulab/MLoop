namespace MLoop.Core.Prediction;

/// <summary>
/// RFC 4180 compliant CSV field parser.
/// Handles commas inside quoted fields correctly.
/// </summary>
public static class CsvFieldParser
{
    /// <summary>
    /// Parses a CSV line respecting RFC 4180 quoted fields. A quote opens a quoted field only where a field starts (white space in
    /// front of it is skipped); a quote anywhere else in a field is part of its text, as CsvHelper and ML.NET's TextLoader read it.
    /// Toggling on every quote took <c>12" pipe,11,1</c> for one quoted field - the rest of the line - and every reader built on
    /// this lost the columns after it.
    /// </summary>
    public static string[] ParseFields(string line)
    {
        var fields = new List<string>();
        bool inQuotes = false;
        bool atFieldStart = true;
        int fieldStart = 0;

        for (int i = 0; i < line.Length; i++)
        {
            char c = line[i];
            if (inQuotes)
            {
                if (c != '"')
                    continue;
                if (i + 1 < line.Length && line[i + 1] == '"')
                    i++; // an escaped quote
                else
                    inQuotes = false;
            }
            else if (c == '"' && atFieldStart)
            {
                inQuotes = true;
                atFieldStart = false;
            }
            else if (c == ',')
            {
                fields.Add(UnquoteField(line[fieldStart..i]));
                fieldStart = i + 1;
                atFieldStart = true;
            }
            else if (!char.IsWhiteSpace(c))
            {
                atFieldStart = false;
            }
        }

        fields.Add(UnquoteField(line[fieldStart..]));
        return fields.ToArray();
    }

    /// <summary>
    /// Formats fields into a RFC 4180 compliant CSV line.
    /// Fields containing commas, quotes, or newlines are quoted.
    /// </summary>
    public static string FormatLine(IEnumerable<string> fields)
    {
        return string.Join(",", fields.Select(Quote));
    }

    /// <summary>
    /// One field as RFC 4180 writes it: quoted, with its quotes doubled, when it holds a comma, a quote or a
    /// line break; as it is otherwise. The one rule every CSV writer in MLoop uses — copies of it disagreed
    /// on line breaks, and a field holding one split its row in two.
    /// </summary>
    public static string Quote(string field)
    {
        if (field.Contains(',') || field.Contains('"') || field.Contains('\n') || field.Contains('\r'))
        {
            return "\"" + field.Replace("\"", "\"\"") + "\"";
        }
        return field;
    }

    private static string UnquoteField(string field)
    {
        field = field.Trim();
        if (field.Length >= 2 && field[0] == '"' && field[^1] == '"')
        {
            field = field[1..^1].Replace("\"\"", "\"");
        }
        return field;
    }
}
