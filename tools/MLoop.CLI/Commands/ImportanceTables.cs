using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>The second ranking of an importance report: the categorical columns, beside the numeric one (a different estimator — never merged into it).</summary>
internal static class ImportanceTables
{
    public static void WriteCategorical(string? method, IReadOnlyList<AnalyzeJson.RankedFeature> ranking)
    {
        if (ranking.Count == 0) return;
        AnsiConsole.WriteLine();
        var table = new Table().Border(TableBorder.Rounded);
        table.AddColumn("Categorical column");
        table.AddColumn(new TableColumn($"Importance ({method})").RightAligned());
        foreach (var r in ranking)
            table.AddRow(Markup.Escape(r.Feature), r.Score.ToString("F4"));
        AnsiConsole.Write(table);
        AnsiConsole.MarkupLine(method == AnalyzeJson.CategoricalRatioMethod
            ? "[grey]The share of the label's variance each column's values explain — its own scale, not comparable with the numeric scores above.[/]"
            : "[grey]Mutual information with the label on each column's own values — its own scale, not comparable with the numeric scores above.[/]");
    }
}
