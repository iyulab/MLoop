namespace MLoop.CLI.Infrastructure.ML;

/// <summary>
/// RFC 4180 compliant CSV field parser.
/// Handles commas inside quoted fields correctly.
/// </summary>
internal static class CsvFieldParser
{
    /// <summary>
    /// Parses a CSV line respecting RFC 4180 quoted fields.
    /// </summary>
    public static string[] ParseFields(string line)
    {
        var fields = new List<string>();
        bool inQuotes = false;
        int fieldStart = 0;

        for (int i = 0; i < line.Length; i++)
        {
            char c = line[i];
            if (c == '"')
            {
                inQuotes = !inQuotes;
            }
            else if (c == ',' && !inQuotes)
            {
                fields.Add(UnquoteField(line[fieldStart..i]));
                fieldStart = i + 1;
            }
        }

        fields.Add(UnquoteField(line[fieldStart..]));
        return fields.ToArray();
    }

    /// <summary>
    /// Formats fields into a RFC 4180 compliant CSV line.
    /// Fields containing commas, quotes, or newlines are quoted.
    /// </summary>
    public static string FormatLine(string[] fields)
    {
        return string.Join(",", fields.Select(QuoteField));
    }

    private static string QuoteField(string field)
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
