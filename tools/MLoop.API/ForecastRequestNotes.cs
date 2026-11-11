using System.Text.Json;

namespace MLoop.API;

/// <summary>
/// What a <c>POST /predict</c> body for a forecasting model asks that the forecast cannot do. A forecast replays the series kept
/// beside the model; the only thing a request can set is <c>horizon</c>. A body that carries a row, rows or other fields used to be
/// answered with the stored series' forecast and no word — a client that sent its own recent points got a confident forecast of
/// another series.
/// </summary>
public static class ForecastRequestNotes
{
    private const int NamesShown = 5;

    /// <summary>The warnings for a request body — empty when it holds nothing the forecast ignores.</summary>
    public static List<string> Of(JsonElement input)
    {
        var notes = new List<string>();
        switch (input.ValueKind)
        {
            case JsonValueKind.Object:
                var ignored = input.EnumerateObject()
                    .Select(p => p.Name)
                    .Where(name => !string.Equals(name, "horizon", StringComparison.Ordinal))
                    .ToList();
                if (ignored.Count > 0)
                    notes.Add(Unused($"the field{(ignored.Count == 1 ? "" : "s")} {Names(ignored)} of the request {(ignored.Count == 1 ? "was" : "were")} not used"));
                if (input.TryGetProperty("horizon", out var horizon) && horizon.ValueKind != JsonValueKind.Number)
                    notes.Add("\"horizon\" must be a number, so it was not used: the model's trained horizon applies.");
                break;
            case JsonValueKind.Array when input.GetArrayLength() > 0:
                notes.Add(Unused($"the {input.GetArrayLength()} row{(input.GetArrayLength() == 1 ? "" : "s")} of the request {(input.GetArrayLength() == 1 ? "was" : "were")} not used"));
                break;
        }
        return notes;
    }

    private static string Unused(string what) =>
        $"A forecast is read from the series kept with the model, not from the request: {what}. " +
        "To forecast another series, train a model on it; the request can set only \"horizon\".";

    private static string Names(List<string> names) =>
        string.Join(", ", names.Take(NamesShown).Select(n => $"\"{n}\"")) + (names.Count > NamesShown ? $" and {names.Count - NamesShown} more" : "");
}
