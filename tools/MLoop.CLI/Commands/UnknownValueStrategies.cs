using static MLoop.Core.Prediction.CategoricalMapper;

namespace MLoop.CLI.Commands;

/// <summary>
/// The names <c>mloop predict --unknown-strategy</c> accepts, each with the strategy it means, in one table. The option was read with
/// a switch whose last arm was <c>auto</c>, so any other word ran as <c>auto</c>: <c>--unknown-strategy errror</c> — a typo by
/// someone who wanted a value training never saw to stop the run — predicted past it, reading the value as missing. A name that is not in
/// the table is refused before the model is loaded.
/// </summary>
internal static class UnknownValueStrategies
{
    private static readonly (string Name, UnknownValueStrategy Strategy)[] Table =
    [
        ("auto", UnknownValueStrategy.Auto),
        ("error", UnknownValueStrategy.Error),
        ("use-most-frequent", UnknownValueStrategy.UseMostFrequent),
        ("use-missing", UnknownValueStrategy.UseMissing),
    ];

    public static IReadOnlyList<string> Names { get; } = [.. Table.Select(entry => entry.Name)];

    /// <summary>The strategy a name means (any case, surrounding space ignored); null for none given or an unknown name.</summary>
    public static UnknownValueStrategy? Parse(string? strategy)
    {
        if (string.IsNullOrWhiteSpace(strategy))
            return null;
        var wanted = strategy.Trim();
        foreach (var (name, meaning) in Table)
            if (string.Equals(name, wanted, StringComparison.OrdinalIgnoreCase))
                return meaning;
        return null;
    }

    /// <summary>The sentence that refuses <paramref name="strategy"/>, or null when it is none or a known one.</summary>
    public static string? Refusal(string? strategy)
        => string.IsNullOrWhiteSpace(strategy) || Parse(strategy) is not null
            ? null
            : $"Unknown strategy '{strategy.Trim()}' for --unknown-strategy. Choose one of: {string.Join(", ", Names)}.";
}
