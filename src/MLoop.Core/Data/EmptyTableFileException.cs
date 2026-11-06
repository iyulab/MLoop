namespace MLoop.Core.Data;

/// <summary>
/// A workbook, Parquet, JSON or other data file converted to a table that has nothing in it - no header row and no data. Raised by
/// <see cref="TabularDataFile.AsCsvAsync"/>, once, with the name of the file the user gave: left to the readers after it, every
/// command met the same empty text in its own way and most said what the CSV parser's state was ("No header record was found",
/// reader and parser offsets) or named the scratch copy.
/// </summary>
public sealed class EmptyTableFileException(string message) : InvalidOperationException(message)
{
}
