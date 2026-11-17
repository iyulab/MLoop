namespace MLoop.Core.Export;

/// <summary>
/// The pairs a caller can have written into an export's manifest (<see cref="OnnxManifest.Annotations"/>): opaque text, bounded,
/// so a manifest stays a file a person reads and a pair cannot be a way to move a payload.
/// </summary>
public static class OnnxAnnotations
{
    /// <summary>The most pairs one manifest carries.</summary>
    public const int MaxPairs = 32;

    /// <summary>The longest key, in characters.</summary>
    public const int MaxKeyLength = 64;

    /// <summary>The longest value, in characters.</summary>
    public const int MaxValueLength = 1024;

    /// <summary>Why <paramref name="annotations"/> cannot be written, or <c>null</c> when they can (or there are none).</summary>
    public static string? Refusal(IReadOnlyDictionary<string, string>? annotations)
    {
        if (annotations is not { Count: > 0 })
            return null;
        if (annotations.Count > MaxPairs)
            return $"annotations: {annotations.Count} pairs given; a manifest carries at most {MaxPairs}.";
        foreach (var (key, value) in annotations)
        {
            if (string.IsNullOrWhiteSpace(key))
                return "annotations: a key is empty.";
            if (key.Length > MaxKeyLength)
                return $"annotations: key '{key[..MaxKeyLength]}…' is longer than {MaxKeyLength} characters.";
            if (key.Any(char.IsControl))
                return $"annotations: key '{key}' holds a control character.";
            if (value is null)
                return $"annotations: '{key}' has no value; give text (it may be empty).";
            if (value.Length > MaxValueLength)
                return $"annotations: the value of '{key}' is longer than {MaxValueLength} characters.";
        }
        return null;
    }

    /// <summary>The pairs in key order (ordinal), so the same pairs always write the same bytes; null when there are none.</summary>
    public static IReadOnlyDictionary<string, string>? Normalize(IReadOnlyDictionary<string, string>? annotations)
        => annotations is { Count: > 0 }
            ? new SortedDictionary<string, string>(annotations.ToDictionary(p => p.Key, p => p.Value), StringComparer.Ordinal)
            : null;
}
