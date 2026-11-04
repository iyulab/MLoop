using CsvHelper;

namespace MLoop.Core.Data;

/// <summary>
/// How MLoop reads and writes a delimited table with the CSV library, in one place. The library's own defaults are not
/// ours: it detects the separator from the first lines (a comma file here is a comma file), recovers from a malformed quote
/// instead of refusing it, and refuses a record of a different width than the first when writing. Every reader and writer
/// of a table takes its options from here so that a file means one thing wherever it is read.
/// </summary>
public static class CsvDialect
{
    /// <summary>
    /// Options for reading a table. The separator is the one given (a comma unless the caller knows better), never detected;
    /// a short row reads its missing cells as empty.
    /// </summary>
    /// <param name="hasHeader">Whether the first record is a header. Without one, the reader moves record by record and takes nothing for a header.</param>
    /// <param name="separator">The field separator.</param>
    /// <param name="trim">Remove spaces and tabs around a value (inside quotes they are kept).</param>
    /// <param name="refuseMalformedQuotes">Refuse a malformed quote. Otherwise the reader recovers: a bare quote is text, and a quote that never closes holds the rest of the file.</param>
    public static CsvReaderOptions Reader(bool hasHeader = true, char separator = ',', bool trim = false, bool refuseMalformedQuotes = false)
    {
        return new CsvReaderOptions
        {
            HasHeader = hasHeader,
            Separator = separator,
            TrimWhitespace = trim,
            MalformedData = refuseMalformedQuotes ? CsvMalformedDataHandling.Throw : CsvMalformedDataHandling.Recover,
        };
    }

    /// <summary>
    /// Options for writing a table: comma separated, a value quoted when it needs it or when it starts or ends with a space or
    /// a tab, and a record of any width accepted — the rows written are the rows given, as they are.
    /// </summary>
    public static CsvWriterOptions Writer(char separator = ',')
    {
        return new CsvWriterOptions
        {
            Separator = separator,
            Quoting = CsvQuoting.WhenNeededOrWhitespace,
            ValidateFieldCount = false,
        };
    }

    /// <summary>The cells of the record the reader stands on, as written (a short record has fewer cells than the header).</summary>
    public static string[] Record(CsvReader csv)
    {
        ArgumentNullException.ThrowIfNull(csv);
        var cells = new string[csv.FieldCount];
        for (var i = 0; i < cells.Length; i++)
            cells[i] = csv.GetString(i);
        return cells;
    }
}
