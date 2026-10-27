using System.Collections.Concurrent;
using MLoop.Core.Embedding;

namespace MLoop.Core.ImageAnomaly;

/// <summary>One image's answer: how unlike the normal images it is, and where in the image.</summary>
/// <param name="Score">The largest of the patches' nearest distances to the bank.</param>
/// <param name="Distances">Each patch's nearest distance, row by row over the grid.</param>
/// <param name="Crop">The part of the image the grid covers, in the image's own pixels.</param>
public sealed record ImageScore(double Score, float[] Distances, ImageRegion Crop);

/// <summary>
/// An image-anomaly model ready to score: its bank and record, and the encoder that made the bank's vectors. It opens
/// only with the encoder the bank was built with — another one's distances mean nothing — and says what differs when
/// that is not the one installed.
/// </summary>
public sealed class ImageAnomalyScorer
{
    private static readonly ConcurrentDictionary<string, (DateTime BankTime, long BankLength, DateTime RecordTime, ImageAnomalyScorer Scorer)> Opened = new();

    private readonly ImageEmbedder _embedder;

    private ImageAnomalyScorer(ImageAnomalyModel model, PatchBank bank, ImageEmbedder embedder, EmbedderPool pool)
    {
        Model = model;
        Bank = bank;
        _embedder = embedder;
        Pool = pool;
    }

    /// <summary>The encoders this scorer's encoder was opened from.</summary>
    internal EmbedderPool Pool { get; }

    /// <summary>The record beside the bank.</summary>
    public ImageAnomalyModel Model { get; }

    /// <summary>The normal images' patch vectors.</summary>
    public PatchBank Bank { get; }

    /// <summary>Whether the model at <paramref name="modelPath"/> is an image-anomaly one.</summary>
    public static bool IsImageAnomaly(string modelPath) => File.Exists(ImageAnomalyModel.PathBeside(modelPath));

    /// <summary>
    /// The scorer for the model at <paramref name="modelPath"/> (its <c>model.zip</c>), or why there is none: not an image-anomaly
    /// model, the bank missing, the encoder not installed or not the bank's. The bank is read once and kept while its files
    /// stay as they are — a promotion that replaces them reads it anew.
    /// </summary>
    public static (ImageAnomalyScorer? Scorer, string? Refusal) Open(string modelPath, EmbedderPool? pool = null)
    {
        var recordPath = ImageAnomalyModel.PathBeside(modelPath);
        var bankPath = ImageAnomalyModel.BankPathBeside(modelPath);
        if (!File.Exists(recordPath))
            return (null, $"'{modelPath}' is not an image-anomaly model: no {Storage.ExperimentLayout.ImageAnomalyFileName} beside it.");
        if (!File.Exists(bankPath))
            return (null, $"The patch bank {Storage.ExperimentLayout.PatchBankFileName} is missing beside '{modelPath}'.");

        pool ??= EmbedderPool.Shared;
        // Kept per model file and per encoder store: a scorer holds the encoder it opened, and the bank is large.
        var key = $"{pool.Store.Root}|{Path.GetFullPath(modelPath)}";
        var bankInfo = new FileInfo(bankPath);
        var recordTime = File.GetLastWriteTimeUtc(recordPath);
        if (Opened.TryGetValue(key, out var held)
            && held.BankTime == bankInfo.LastWriteTimeUtc && held.BankLength == bankInfo.Length && held.RecordTime == recordTime
            && ReferenceEquals(held.Scorer.Pool, pool))
            return (held.Scorer, null);

        var model = ImageAnomalyModel.ReadBeside(modelPath)!;
        var (embedder, refusal) = pool.Get(model.Encoder.Id);
        if (embedder is null)
            return (null, $"The bank was built with encoder '{model.Encoder.Id}': {refusal}");
        if (model.Mismatch(EncoderIdentity.Of(embedder.Encoder)) is { } mismatch)
            return (null, $"This model cannot score with the installed encoder: {mismatch}. Train it again, or install the encoder it was built with.");

        PatchBank bank;
        try
        {
            bank = PatchBank.Read(bankPath);
        }
        catch (InvalidDataException ex)
        {
            return (null, ex.Message);
        }
        if (bank.Dimension != model.Encoder.Dimension)
            return (null, $"The bank holds vectors of {bank.Dimension} numbers, the encoder's are {model.Encoder.Dimension}.");

        var scorer = new ImageAnomalyScorer(model, bank, embedder, pool);
        Opened[key] = (bankInfo.LastWriteTimeUtc, bankInfo.Length, recordTime, scorer);
        return (scorer, null);
    }

    /// <summary>The images' answers, in order.</summary>
    /// <exception cref="UnreadableImageException">An image could not be decoded.</exception>
    public IReadOnlyList<ImageScore> Score(IReadOnlyList<byte[]> images, CancellationToken cancellationToken = default)
    {
        var scores = new List<ImageScore>(images.Count);
        foreach (var batch in images.Chunk(EmbedderPool.MaxPatchImages))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var embedded = _embedder.Embed(batch, patches: true);
            foreach (var image in embedded.Patches!)
            {
                var (score, distances) = PatchBankScorer.Score(Bank, image.Vectors);
                scores.Add(new ImageScore(score, distances, image.Crop));
            }
        }
        return scores;
    }
}
