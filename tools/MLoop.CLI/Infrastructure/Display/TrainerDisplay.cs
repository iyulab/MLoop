using MLoop.Core.Models;

namespace MLoop.CLI.Infrastructure.Display;

/// <summary>
/// How a trainer's name is shortened for a column, and where the parts that do not fit go.
/// </summary>
/// <remarks>
/// <para>
/// What the core records is the whole pipeline AutoML assembled —
/// <c>ReplaceMissingValues=&gt;Concatenate=&gt;LightGbmBinary</c> — and it is right to record all of
/// it: the transforms are part of what produced the model. But a column headed "Trainer" that
/// prints a pipeline is showing the wrong thing at the wrong width, and in an eighty-column
/// terminal that value wraps across five lines, pushing every other cell of the row down with it.
/// </para>
/// <para>
/// The trainer is the last stage; everything before it is preprocessing AutoML chose. So the column
/// shows the last stage, and the full chain stays available underneath the table for the reader who
/// wants to know what ran. Shortening belongs here rather than in the core, which transmits the
/// whole name and lets each consumer decide how much of it it has room for.
/// </para>
/// </remarks>
internal static class TrainerDisplay
{
    private const string ChainSeparator = "=>";

    /// <summary>
    /// The trainer alone, without the transforms AutoML put in front of it.
    /// </summary>
    /// <remarks>
    /// A name with no chain in it is already the trainer and comes back unchanged, which is the
    /// ordinary case — the chain appears only when AutoML assembled one.
    /// </remarks>
    internal static string Short(string? fullName)
    {
        if (string.IsNullOrWhiteSpace(fullName))
            return "-";

        var lastStage = fullName.LastIndexOf(ChainSeparator, StringComparison.Ordinal);
        return lastStage < 0
            ? fullName.Trim()
            : fullName[(lastStage + ChainSeparator.Length)..].Trim();
    }

    /// <summary>
    /// What the Trainer column shows: the trainer without the transforms in front of it, still
    /// carrying the hyperparameters that tell one configuration from another.
    /// </summary>
    /// <remarks>
    /// The parameters are not decoration here — a leaderboard exists to compare runs of the same
    /// trainer, so <c>KMeans (k=2)</c> and <c>KMeans (k=3)</c> are two different rows and dropping
    /// what is inside the parentheses makes the table say the same thing twice. Only the chain
    /// prefix is removable, because it is the same on every row. The fallback reason is not here: it
    /// is a sentence, and it goes under the table.
    /// </remarks>
    internal static string Short(TrainerDescriptor? trainer)
    {
        if (trainer is null)
            return "-";

        var name = Short(trainer.Name);
        return trainer.Parameters is { Count: > 0 }
            ? $"{name} ({string.Join(", ", trainer.Parameters.Select(p => $"{p.Key}={p.Value}"))})"
            : name;
    }

    /// <summary>Whether <paramref name="fullName"/> had stages the column is not showing.</summary>
    internal static bool WasShortened(string? fullName) =>
        !string.IsNullOrWhiteSpace(fullName) && fullName.Contains(ChainSeparator, StringComparison.Ordinal);

    /// <summary>
    /// The line printed under a table for one row whose trainer did not fit, or whose trainer stood
    /// in for the one the run intended.
    /// </summary>
    /// <remarks>
    /// Written from the descriptor rather than from <see cref="TrainerDescriptor.Display"/>: the
    /// reason is a field, and recovering it by taking a display string apart is how a change of
    /// wording becomes a change of behaviour — the same thing the descriptor's own documentation
    /// warns against.
    /// </remarks>
    internal static string? Footnote(string marker, TrainerDescriptor? trainer)
    {
        if (trainer is null)
            return null;

        var parts = new List<string>(2);
        if (WasShortened(trainer.Name))
            parts.Add(trainer.Name);
        if (!string.IsNullOrEmpty(trainer.FallbackReason))
            parts.Add(trainer.FallbackReason);

        return parts.Count == 0 ? null : $"{marker} {string.Join(" — ", parts)}";
    }
}
