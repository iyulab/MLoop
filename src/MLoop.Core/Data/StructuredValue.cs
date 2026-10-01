using System.Text.Json;

namespace MLoop.Core.Data;

/// <summary>
/// Whether a cell holds a JSON list or object rather than one value — how a nested column (a
/// Parquet list, a JSON array) is carried in a table.
/// </summary>
public static class StructuredValue
{
    /// <summary>
    /// True when <paramref name="value"/> parses as a JSON array or object. Text that merely starts
    /// with a bracket (<c>[draft] notes</c>) is not.
    /// </summary>
    public static bool Is(string value)
    {
        var trimmed = value.AsSpan().Trim();
        if (trimmed.Length < 2) return false;
        var (open, close) = (trimmed[0], trimmed[^1]);
        if (!(open == '[' && close == ']') && !(open == '{' && close == '}'))
            return false;

        try
        {
            using var document = JsonDocument.Parse(value);
            return document.RootElement.ValueKind is JsonValueKind.Array or JsonValueKind.Object;
        }
        catch (JsonException)
        {
            return false;
        }
    }
}
