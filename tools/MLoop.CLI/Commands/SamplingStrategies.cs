namespace MLoop.CLI.Commands;

/// <summary>
/// The names <c>mloop train --sampling-strategy</c> accepts, in one place. Sampling reads the name with a
/// two-way branch, so any other word used to fall through to random sampling: <c>--sampling-strategy stratifed</c>
/// (a typo) ran, said "random" in its output line, and left the user with a sample that ignored the classes they
/// had asked it to keep. A name that is not one of these is refused where the option is read, and again where the
/// sampling is done, so no caller can bring the silent fallback back.
/// </summary>
internal static class SamplingStrategies
{
    public const string Random = "random";
    public const string Stratified = "stratified";

    public static IReadOnlyList<string> Names { get; } = [Random, Stratified];

    /// <summary>The strategy as sampling compares it (any case, surrounding space ignored); null when none was given.</summary>
    public static string? Normalize(string? strategy)
    {
        if (string.IsNullOrWhiteSpace(strategy))
            return null;
        return strategy.Trim().ToLowerInvariant();
    }

    /// <summary>The sentence that refuses <paramref name="strategy"/>, or null when it is none or a known one.</summary>
    public static string? Refusal(string? strategy)
    {
        var normalized = Normalize(strategy);
        return normalized is null || Names.Contains(normalized)
            ? null
            : $"Unknown sampling strategy '{strategy!.Trim()}'. Choose one of: {string.Join(", ", Names)}.";
    }

    public const string RefusalTip = "Leave --sampling-strategy out to sample by task: stratified for classification, random otherwise.";
}
