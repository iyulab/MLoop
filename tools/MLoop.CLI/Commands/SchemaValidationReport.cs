using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.ML;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// How <c>predict</c> and <c>evaluate</c> report data that failed the saved-schema check: the cause and its
/// suggestions as one <c>Error</c>/<c>Tip</c> pair on stderr — which a <c>--json</c> consumer receives as its
/// document — then the missing columns one per line. One report, so the two commands cannot drift apart.
/// </summary>
internal static class SchemaValidationReport
{
    /// <param name="result">The failed check.</param>
    /// <param name="dataNoun">What was checked, for the column list: "prediction data", "test data".</param>
    public static void Refuse(SchemaValidationResult result, string dataNoun)
    {
        ArgumentNullException.ThrowIfNull(result);
        var cause = "Schema validation failed.";
        if (!string.IsNullOrEmpty(result.ErrorMessage))
            cause += " " + result.ErrorMessage;
        // Only when the message did not already name them: it usually does, and the list below has the room to.
        else if (result.MissingColumns.Count > 0)
            cause += " Missing columns: " + string.Join(", ", result.MissingColumns);

        if (result.Suggestions.Count > 0)
            ErrorConsole.Error(Markup.Escape(cause), Markup.Escape(string.Join(" ", result.Suggestions)));
        else
            ErrorConsole.Error(Markup.Escape(cause));

        if (result.MissingColumns.Count == 0)
            return;
        var err = ErrorConsole.Out;
        err.MarkupLine($"Missing columns in {Markup.Escape(dataNoun)}:");
        foreach (var column in result.MissingColumns)
            err.MarkupLine($"  [grey]-[/] {Markup.Escape(column)}");
    }
}
