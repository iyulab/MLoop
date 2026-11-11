namespace MLoop.Core.Data;

/// <summary>
/// A time series with a missing value in the column it is read from (<see cref="MissingNumbers.RefuseGapsInSeries"/>). A type of its
/// own so the command that reports it can tell it from a missing column: the message names the series column and the
/// <c>--columns</c> option of the fill that mends it, and a keyword rule reading "column" in that text answered with advice about
/// column names - the wrong thing to check in a file whose columns are fine.
/// </summary>
public sealed class SeriesGapException(string message) : InvalidOperationException(message)
{
}
