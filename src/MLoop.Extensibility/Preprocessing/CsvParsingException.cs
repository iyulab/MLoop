namespace MLoop.Extensibility.Preprocessing;

/// <summary>
/// Exception thrown when CSV parsing fails.
/// </summary>
public class CsvParsingException : Exception
{
    /// <summary>
    /// Path to CSV file that failed to parse.
    /// </summary>
    public string? FilePath { get; init; }

    /// <summary>
    /// Line number where parsing failed (if available).
    /// </summary>
    public int? LineNumber { get; init; }

    public CsvParsingException(string message) : base(message)
    {
    }

    public CsvParsingException(string message, Exception innerException)
        : base(message, innerException)
    {
    }

    public CsvParsingException(string message, string filePath)
        : base(message)
    {
        FilePath = filePath;
    }

    public CsvParsingException(string message, string filePath, int lineNumber)
        : base(message)
    {
        FilePath = filePath;
        LineNumber = lineNumber;
    }
}
