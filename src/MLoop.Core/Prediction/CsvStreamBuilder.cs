using System.Globalization;
using System.Text;
using MLoop.Core.Models;

namespace MLoop.Core.Prediction;

public static class CsvStreamBuilder
{
    public static MemoryStream Build(
        Dictionary<string, object>[] rows,
        InputSchemaInfo schema,
        bool injectDummyLabel = false)
    {
        var activeColumns = schema.Columns
            .Where(c => !c.Purpose.Equals("Exclude", StringComparison.OrdinalIgnoreCase))
            .ToList();

        var columnNames = activeColumns.Select(c => c.Name).ToList();

        var labelCol = activeColumns.FirstOrDefault(c =>
            c.Purpose.Equals("Label", StringComparison.OrdinalIgnoreCase));
        bool needsDummyLabel = injectDummyLabel && labelCol != null &&
            rows.Length > 0 && !rows[0].ContainsKey(labelCol.Name);

        var stream = new MemoryStream();
        stream.Write(Encoding.UTF8.GetPreamble()); // UTF-8 BOM for ML.NET

        using var writer = new StreamWriter(stream, Encoding.UTF8, leaveOpen: true);

        // Header
        writer.WriteLine(string.Join(",", columnNames));

        // Data rows
        foreach (var row in rows)
        {
            var fields = new string[columnNames.Count];
            for (int i = 0; i < columnNames.Count; i++)
            {
                var colName = columnNames[i];
                if (needsDummyLabel && colName == labelCol!.Name)
                    fields[i] = "0";
                else if (row.TryGetValue(colName, out var value) && value != null)
                    fields[i] = FormatValue(value);
                else
                    fields[i] = "";
            }
            writer.WriteLine(string.Join(",", fields.Select(QuoteCsvField)));
        }

        writer.Flush();
        stream.Position = 0;
        return stream;
    }

    private static string FormatValue(object value) => value switch
    {
        float f => f.ToString(CultureInfo.InvariantCulture),
        double d => d.ToString(CultureInfo.InvariantCulture),
        decimal m => m.ToString(CultureInfo.InvariantCulture),
        int i => i.ToString(CultureInfo.InvariantCulture),
        long l => l.ToString(CultureInfo.InvariantCulture),
        bool b => b ? "1" : "0",
        _ => value.ToString() ?? ""
    };

    private static string QuoteCsvField(string value)
    {
        if (value.Contains(',') || value.Contains('"') || value.Contains('\n') || value.Contains('\r'))
            return "\"" + value.Replace("\"", "\"\"") + "\"";
        return value;
    }
}
