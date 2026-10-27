using System.Text.Json;
using System.Text.Json.Serialization;
using MLoop.Core.Embedding;
using MLoop.Core.Storage;

namespace MLoop.Core.ImageAnomaly;

/// <summary>How far held-out normal images scored: where "normal" ends for this bank, so a score can be read against it.</summary>
/// <param name="Images">How many normal images were held out of the bank and scored.</param>
/// <param name="Median">The median of their scores.</param>
/// <param name="P95">Their 95th percentile (nearest rank).</param>
/// <param name="P99">Their 99th percentile (nearest rank).</param>
/// <param name="Max">The highest.</param>
public sealed record ScoreBaseline(int Images, double Median, double P95, double P99, double Max)
{
    /// <summary>The baseline of <paramref name="scores"/> (nearest-rank percentiles), or <see langword="null"/> when there are none.</summary>
    public static ScoreBaseline? Of(IReadOnlyList<double> scores)
    {
        if (scores.Count == 0) return null;
        var sorted = scores.Order().ToArray();
        double Rank(double p) => sorted[Math.Clamp((int)Math.Ceiling(p * sorted.Length) - 1, 0, sorted.Length - 1)];
        return new ScoreBaseline(sorted.Length, Rank(0.5), Rank(0.95), Rank(0.99), sorted[^1]);
    }
}

/// <summary>
/// The record of an image-anomaly experiment, beside its patch bank: which encoder's vector space the bank is in
/// (a distance between vectors of two spaces means nothing, so scoring refuses a different one), the patch grid, how the
/// bank was chosen, and the baseline of held-out normal scores.
/// </summary>
/// <param name="Encoder">The identity of the encoder that made the bank's vectors.</param>
/// <param name="Grid">The patch grid of one image.</param>
/// <param name="Images">How many normal images the bank was built from.</param>
/// <param name="HeldOutImages">How many normal images were kept out of the bank to be scored against it.</param>
/// <param name="TotalPatches">How many patches those images had.</param>
/// <param name="BankPatches">How many of them the bank holds.</param>
/// <param name="Selection"><c>all</c>, or <c>random</c> when the bank is a seeded subsample of the patches.</param>
/// <param name="Seed">The subsample's seed (null for <c>all</c>).</param>
/// <param name="Baseline">The held-out normal images' scores; null when none were held out.</param>
public sealed record ImageAnomalyModel(
    EncoderIdentity Encoder,
    PatchGrid Grid,
    int Images,
    int HeldOutImages,
    long TotalPatches,
    int BankPatches,
    string Selection,
    int? Seed,
    ScoreBaseline? Baseline)
{
    private static readonly JsonSerializerOptions Options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true,
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    /// <summary>The record beside <paramref name="modelPath"/>, or <see langword="null"/> when the model is not an image-anomaly one.</summary>
    public static ImageAnomalyModel? ReadBeside(string modelPath)
    {
        var path = PathBeside(modelPath);
        return File.Exists(path) ? JsonSerializer.Deserialize<ImageAnomalyModel>(File.ReadAllText(path), Options) : null;
    }

    /// <summary>Writes this record beside <paramref name="modelPath"/>.</summary>
    public void WriteBeside(string modelPath) => File.WriteAllText(PathBeside(modelPath), JsonSerializer.Serialize(this, Options));

    /// <summary>Where the record beside <paramref name="modelPath"/> lives.</summary>
    public static string PathBeside(string modelPath)
        => Path.Combine(Path.GetDirectoryName(Path.GetFullPath(modelPath)) ?? ".", ExperimentLayout.ImageAnomalyFileName);

    /// <summary>Where the patch bank beside <paramref name="modelPath"/> lives.</summary>
    public static string BankPathBeside(string modelPath)
        => Path.Combine(Path.GetDirectoryName(Path.GetFullPath(modelPath)) ?? ".", ExperimentLayout.PatchBankFileName);

    /// <summary>
    /// Why scoring with <paramref name="installed"/> would not mean what training measured, or <see langword="null"/> when the
    /// vector space is the bank's: the same weights, the same preprocessing, the same length.
    /// </summary>
    public string? Mismatch(EncoderIdentity installed)
    {
        var recorded = Encoder;
        if (!string.Equals(recorded.Id, installed.Id, StringComparison.Ordinal))
            return $"the bank was built with encoder '{recorded.Id}', and '{installed.Id}' is the one in use";
        if (!string.Equals(recorded.Sha256, installed.Sha256, StringComparison.OrdinalIgnoreCase))
            return $"encoder '{recorded.Id}' has other weights than the bank was built with (sha256 {Short(recorded.Sha256)} then, {Short(installed.Sha256)} now)";
        if (!string.Equals(recorded.Preprocessor, installed.Preprocessor, StringComparison.Ordinal))
            return $"images are prepared as '{installed.Preprocessor}' now and were '{recorded.Preprocessor}' when the bank was built";
        if (recorded.Dimension != installed.Dimension)
            return $"the encoder's vectors have {installed.Dimension} numbers, the bank's {recorded.Dimension}";
        return null;

        static string Short(string? sha) => string.IsNullOrEmpty(sha) ? "unknown" : sha[..Math.Min(8, sha.Length)];
    }
}
