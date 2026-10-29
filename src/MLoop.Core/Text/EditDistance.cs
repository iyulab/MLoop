namespace MLoop.Core.Text;

/// <summary>
/// How far two spellings are apart — the insertions, deletions and substitutions between them. The one
/// measure behind every "did you mean" in the product, so two suggestions never disagree about what is near.
/// </summary>
public static class EditDistance
{
    public static int Between(string a, string b)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        var previous = new int[b.Length + 1];
        var current = new int[b.Length + 1];
        for (var j = 0; j <= b.Length; j++)
            previous[j] = j;
        for (var i = 1; i <= a.Length; i++)
        {
            current[0] = i;
            for (var j = 1; j <= b.Length; j++)
            {
                var substitution = previous[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1);
                current[j] = Math.Min(substitution, Math.Min(previous[j] + 1, current[j - 1] + 1));
            }
            (previous, current) = (current, previous);
        }
        return previous[b.Length];
    }

    /// <summary>
    /// The candidate nearest to <paramref name="typed"/>, when one is near enough to be what was meant: a name that
    /// begins with what was typed (an abbreviation, three letters or more) or one within two edits and under half the
    /// typed length (a slip of the keys; three edits for a name of twelve letters or more) — otherwise <see langword="null"/>: a far name is not a suggestion.
    /// </summary>
    public static string? Nearest(string typed, IEnumerable<string> candidates)
    {
        ArgumentNullException.ThrowIfNull(typed);
        var names = candidates as IReadOnlyCollection<string> ?? candidates.ToList();
        if (typed.Length >= 3)
        {
            var abbreviated = names.Where(n => n.Length > typed.Length && n.StartsWith(typed, StringComparison.Ordinal)).ToList();
            if (abbreviated.Count == 1)
                return abbreviated[0];
        }
        string? best = null;
        var bestDistance = int.MaxValue;
        foreach (var candidate in names)
        {
            var distance = Between(typed, candidate);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                best = candidate;
            }
        }
        return (bestDistance <= 2 && bestDistance * 2 < typed.Length) || (bestDistance == 3 && typed.Length >= 12) ? best : null;
    }
}
