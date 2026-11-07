namespace MLoop.Core.Data;

/// <summary>
/// A delimited file with a quoted field that opens and never closes: every line after it, to the end of the file, belongs to that one
/// field. Raised by <see cref="CsvDataLoader.FlattenMultiLineQuotedFields"/> with the line the quote opens on and the file the user
/// gave. Before it, the lines were written out as one record - a 40-row file read as 10, trained on 9, and said
/// "Flattened 0 multi-line quoted field(s)".
/// </summary>
public sealed class UnterminatedQuoteException(string message) : InvalidOperationException(message)
{
}
