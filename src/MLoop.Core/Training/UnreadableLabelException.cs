namespace MLoop.Core.Training;

/// <summary>
/// A numeric label with values the loader cannot read (<see cref="NumericLabelCheck"/>), refused before the file is scored. A type of
/// its own so the command that reports it can tell it apart from a missing column and say what to do about the values — a message
/// that mentions a label and a column is otherwise answered with advice about column names.
/// </summary>
public sealed class UnreadableLabelException(string message) : InvalidOperationException(message);
