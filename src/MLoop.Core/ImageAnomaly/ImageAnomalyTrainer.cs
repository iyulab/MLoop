using MLoop.Core.Embedding;

namespace MLoop.Core.ImageAnomaly;

/// <summary>An image-anomaly model as training makes it: the bank and the record that names its vector space.</summary>
/// <param name="Bank">The normal images' patch vectors.</param>
/// <param name="Model">The record written beside the bank.</param>
public sealed record ImageAnomalyArtifacts(PatchBank Bank, ImageAnomalyModel Model)
{
    /// <summary>Writes the bank and its record beside <paramref name="modelPath"/>.</summary>
    public void WriteBeside(string modelPath)
    {
        Bank.Write(ImageAnomalyModel.BankPathBeside(modelPath));
        Model.WriteBeside(modelPath);
    }
}

/// <summary>
/// Builds an image-anomaly model from a folder of normal images: every image's patch vectors from the encoder
/// (<see cref="ImageEmbedder"/>, the code behind <c>POST /embed?patches=true</c>), the whole set or a seeded subsample
/// of it as the bank, and — from images held out of the bank — how far normal images score.
/// </summary>
public static class ImageAnomalyTrainer
{
    /// <summary>The seed of the held-out choice and of a subsampled bank: the same folder gives the same model.</summary>
    public const int Seed = 42;

    /// <summary>The share of images held out of the bank to be scored against it.</summary>
    public const double HeldOutShare = 0.10;

    /// <summary>
    /// Fewer images than this are all kept in the bank: there is no baseline to take from a handful, and a bank of
    /// fewer would be thinner than the few normal images already are.
    /// </summary>
    public const int MinImagesToHoldOut = 10;

    /// <summary>
    /// The most numbers (2 GiB of floats) the patches may come to for a coreset to be chosen from them: the choice needs
    /// every patch at once. A bank limited from more than this is a seeded random subsample, streamed.
    /// </summary>
    internal const long MaxCoresetFloats = 1L << 29;

    private static readonly HashSet<string> Extensions = new(StringComparer.OrdinalIgnoreCase)
        { ".jpg", ".jpeg", ".png", ".bmp", ".gif" };

    /// <summary>The image extensions a folder is read for, for messages that name them.</summary>
    public static string ExtensionList => string.Join(", ", Extensions.Order(StringComparer.Ordinal));

    /// <summary>Every image under <paramref name="directory"/>, at any depth, ordered by path — empty for a missing folder.</summary>
    public static IReadOnlyList<string> ListImages(string directory) =>
        string.IsNullOrWhiteSpace(directory) || !Directory.Exists(directory)
            ? []
            : [.. Directory.EnumerateFiles(directory, "*", SearchOption.AllDirectories)
                .Where(f => Extensions.Contains(Path.GetExtension(f)))
                .Select(Path.GetFullPath)
                .OrderBy(p => p, StringComparer.Ordinal)];

    /// <summary>
    /// The bank of the normal images under <paramref name="directory"/>.
    /// </summary>
    /// <param name="directory">The folder of normal images.</param>
    /// <param name="encoderId">The image encoder (null: the only one installed).</param>
    /// <param name="bankSize">
    /// The most patches the bank holds; null keeps every patch of the images in it. A limit below the patches the images have
    /// is met with a coreset (<see cref="CoresetSelector"/>), or with a seeded random subsample when the patches are too many
    /// to hold at once (<see cref="MaxCoresetFloats"/>).
    /// </param>
    /// <param name="log">Where progress goes.</param>
    /// <param name="pool">The encoders (null: the process's).</param>
    /// <param name="cancellationToken">Stops the embedding between batches.</param>
    /// <exception cref="InvalidOperationException">No images, no usable encoder, or an image that cannot be decoded.</exception>
    public static (ImageAnomalyArtifacts Artifacts, Dictionary<string, double> Metrics) Train(
        string directory, string? encoderId, int? bankSize, Action<string>? log = null,
        EmbedderPool? pool = null, CancellationToken cancellationToken = default)
        => Train(directory, encoderId, bankSize, MaxCoresetFloats, log, pool, cancellationToken);

    /// <summary><see cref="Train(string, string?, int?, Action{string}?, EmbedderPool?, CancellationToken)"/> with the memory a coreset may take as a parameter.</summary>
    internal static (ImageAnomalyArtifacts Artifacts, Dictionary<string, double> Metrics) Train(
        string directory, string? encoderId, int? bankSize, long maxCoresetFloats, Action<string>? log,
        EmbedderPool? pool, CancellationToken cancellationToken)
    {
        var say = log ?? (_ => { });
        var images = ListImages(directory);
        if (images.Count == 0)
            throw new InvalidOperationException(
                $"No images in '{directory}' ({ExtensionList}). An image-anomaly model learns from a folder of normal images.");
        if (bankSize is <= 0)
            throw new InvalidOperationException("--bank-size is the most patches the bank holds, at least 1.");

        var (embedder, refusal) = (pool ?? EmbedderPool.Shared).Get(encoderId);
        if (embedder is null)
            throw new InvalidOperationException(refusal);
        if (ImageEmbedder.NoPatches(embedder.Encoder) is { } noPatches)
            throw new InvalidOperationException(noPatches);
        var grid = embedder.Encoder.Grid!;
        var identity = EncoderIdentity.Of(embedder.Encoder);
        var perImage = grid.Rows * grid.Columns;

        // The held-out images are a seeded choice, so the same folder gives the same bank and the same baseline.
        var order = Enumerable.Range(0, images.Count).ToArray();
        var random = new Random(Seed);
        for (var i = order.Length - 1; i > 0; i--)
        {
            var j = random.Next(i + 1);
            (order[i], order[j]) = (order[j], order[i]);
        }
        var heldOutCount = images.Count >= MinImagesToHoldOut ? Math.Max(1, (int)(images.Count * HeldOutShare)) : 0;
        var heldOut = order.Take(heldOutCount).Order().Select(i => images[i]).ToList();
        var bankImages = order.Skip(heldOutCount).Order().Select(i => images[i]).ToList();

        var total = (long)bankImages.Count * perImage;
        var limited = bankSize is { } cap && cap < total;
        var coreset = limited && total * identity.Dimension <= maxCoresetFloats;
        var keep = limited && !coreset ? PickPatches(total, bankSize!.Value, new Random(Seed)) : null;
        // A coreset is chosen from every patch, so they are all held until the choice; a random subsample is taken as they stream by.
        var selected = coreset ? (int)total : keep?.Length ?? (int)Math.Min(total, int.MaxValue / identity.Dimension);
        if (!limited && total > selected)
            throw new InvalidOperationException(
                $"{total:N0} patches of {identity.Dimension} numbers are more than one bank holds; give --bank-size.");

        say($"Embedding {bankImages.Count} images with '{identity.Id}' ({perImage} patches each)" +
            (limited ? $", keeping {bankSize:N0} of {total:N0} patches by {(coreset ? "a coreset" : "a seeded random subsample")} (seed {Seed})." : "."));
        if (limited && !coreset)
            say($"  {total:N0} patches are too many to choose a coreset from in memory; the {bankSize:N0} are a random subsample.");
        var data = new float[checked(selected * identity.Dimension)];
        var taken = 0;
        var seen = 0L;
        var nextPick = 0;
        var done = 0;
        foreach (var batch in bankImages.Chunk(EmbedderPool.MaxPatchImages))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var embedded = EmbedBatch(embedder, batch);
            foreach (var image in embedded.Patches!)
            {
                foreach (var vector in image.Vectors)
                {
                    if (keep is null || (nextPick < keep.Length && keep[nextPick] == seen))
                    {
                        vector.CopyTo(data, taken * identity.Dimension);
                        taken++;
                        nextPick++;
                    }
                    seen++;
                }
            }
            done += batch.Length;
            if (done == batch.Length || done % 80 < batch.Length || done == bankImages.Count)
                say($"  {done}/{bankImages.Count} images");
        }

        if (coreset)
        {
            say($"Choosing {bankSize:N0} of {total:N0} patches (greedy k-center).");
            var rows = CoresetSelector.Select(data, identity.Dimension, bankSize!.Value, Seed, cancellationToken);
            var kept = new float[checked(rows.Length * identity.Dimension)];
            for (var i = 0; i < rows.Length; i++)
                Array.Copy(data, (long)rows[i] * identity.Dimension, kept, (long)i * identity.Dimension, identity.Dimension);
            data = kept;
        }

        var bank = PatchBank.FromRows(data, identity.Dimension);
        var scores = new List<double>();
        if (heldOut.Count > 0)
        {
            say($"Scoring {heldOut.Count} held-out normal images against the bank.");
            foreach (var batch in heldOut.Chunk(EmbedderPool.MaxPatchImages))
            {
                cancellationToken.ThrowIfCancellationRequested();
                foreach (var image in EmbedBatch(embedder, batch).Patches!)
                    scores.Add(PatchBankScorer.Score(bank, image.Vectors).Score);
            }
        }
        else
        {
            say($"Fewer than {MinImagesToHoldOut} images: all stay in the bank, and no baseline of normal scores is taken.");
        }

        var baseline = ScoreBaseline.Of(scores);
        var record = new ImageAnomalyModel(identity, grid, bankImages.Count, heldOut.Count, total, bank.Count,
            !limited ? "all" : coreset ? "coreset" : "random", !limited ? null : Seed, baseline);
        var metrics = new Dictionary<string, double>
        {
            ["images"] = bankImages.Count + heldOut.Count,
            ["bank_patches"] = bank.Count,
        };
        if (baseline is not null)
        {
            metrics["held_out_images"] = baseline.Images;
            metrics["normal_score_median"] = baseline.Median;
            metrics["normal_score_p95"] = baseline.P95;
            metrics["normal_score_p99"] = baseline.P99;
            metrics["normal_score_max"] = baseline.Max;
        }
        return (new ImageAnomalyArtifacts(bank, record), metrics);
    }

    private static EmbeddingResult EmbedBatch(ImageEmbedder embedder, string[] files)
    {
        var bytes = new List<byte[]>(files.Length);
        foreach (var file in files)
            bytes.Add(File.ReadAllBytes(file));
        try
        {
            return embedder.Embed(bytes, patches: true);
        }
        catch (UnreadableImageException ex)
        {
            throw new InvalidOperationException($"'{files[ex.Index]}' could not be read as an image: {ex.Message}", ex);
        }
    }

    /// <summary>Which <paramref name="count"/> of <paramref name="total"/> patches a subsampled bank keeps — seeded, without replacement, ascending.</summary>
    internal static long[] PickPatches(long total, int count, Random random)
    {
        // Floyd's sampling: count draws, no table of the whole range.
        var chosen = new HashSet<long>(count);
        for (var j = total - count; j < total; j++)
        {
            var t = random.NextInt64(j + 1);
            if (!chosen.Add(t)) chosen.Add(j);
        }
        var picked = chosen.ToArray();
        Array.Sort(picked);
        return picked;
    }
}
