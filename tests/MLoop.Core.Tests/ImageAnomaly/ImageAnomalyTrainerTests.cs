using MLoop.Core.Embedding;
using MLoop.Core.ImageAnomaly;
using SkiaSharp;

namespace MLoop.Core.Tests.ImageAnomaly;

/// <summary>
/// A folder of normal images becomes a bank of their patch vectors, with the encoder's identity beside it and, from images
/// held out of the bank, a baseline of normal scores — on the four-dimensional patch encoder graph the embedding tests use.
/// </summary>
public class ImageAnomalyTrainerTests : IDisposable
{
    private static readonly string Fixtures = Path.Combine(AppContext.BaseDirectory, "Fixtures", "embedding");
    private readonly string _root = Directory.CreateTempSubdirectory("mloop-anomaly-").FullName;
    private readonly string _images;
    private readonly EmbedderPool _pool;

    // A 2 × 2 grid of 3-pixel patches over a 6-pixel crop.
    private static readonly EncoderDefinition TinyPatches = new(
        "tiny", "1", Dimension: 4, ImagePreprocessing.ImageNet(shortestEdge: 8, cropSize: 6),
        InputName: "pixel_values", OutputName: "last_hidden_state", EncoderPooling.FirstToken, PatchSize: 3);

    public ImageAnomalyTrainerTests()
    {
        _images = Path.Combine(_root, "normal");
        Directory.CreateDirectory(_images);
        var store = new EncoderStore(Path.Combine(_root, "encoders"));
        store.Add(TinyPatches, Path.Combine(Fixtures, "tiny-patch-encoder.onnx"));
        _pool = new EmbedderPool(store);
    }

    public void Dispose()
    {
        _pool.Dispose();
        try { Directory.Delete(_root, recursive: true); } catch { /* temp */ }
    }

    private void WriteImages(int count, string? folder = null)
    {
        var dir = folder ?? _images;
        Directory.CreateDirectory(dir);
        for (var i = 0; i < count; i++)
        {
            using var bitmap = new SKBitmap(12, 12);
            for (var y = 0; y < 12; y++)
                for (var x = 0; x < 12; x++)
                    bitmap.SetPixel(x, y, new SKColor((byte)((x * 20 + i * 11) % 256), (byte)((y * 18 + i * 7) % 256), (byte)((x + y + i * 13) % 256)));
            using var data = SKImage.FromBitmap(bitmap).Encode(SKEncodedImageFormat.Png, 100);
            File.WriteAllBytes(Path.Combine(dir, $"n{i:000}.png"), data.ToArray());
        }
    }

    private (ImageAnomalyArtifacts Artifacts, Dictionary<string, double> Metrics) Train(int? bankSize = null)
        => ImageAnomalyTrainer.Train(_images, "tiny", bankSize, pool: _pool);

    [Fact]
    public void TheBank_HoldsTheBankImagesPatches_AndTheRestAreHeldOutForTheBaseline()
    {
        WriteImages(20);

        var (artifacts, metrics) = Train();

        // 10 % of 20 held out, the other 18 images × 4 patches in the bank, every patch kept
        Assert.Equal((18, 2), (artifacts.Model.Images, artifacts.Model.HeldOutImages));
        Assert.Equal((72L, 72, "all", (int?)null), (artifacts.Model.TotalPatches, artifacts.Bank.Count, artifacts.Model.Selection, artifacts.Model.Seed));
        Assert.Equal((4, 4), (artifacts.Bank.Dimension, artifacts.Model.Encoder.Dimension));
        Assert.Equal(new PatchGrid(2, 2, 3, 6), artifacts.Model.Grid);
        Assert.Equal("tiny", artifacts.Model.Encoder.Id);
        Assert.Equal(2, artifacts.Model.Baseline!.Images);
        Assert.Equal(20, metrics["images"]);
        Assert.Equal(72, metrics["bank_patches"]);
        Assert.True(metrics["normal_score_max"] >= metrics["normal_score_median"]);
        Assert.All(artifacts.Bank.Data, v => Assert.True(float.IsFinite(v)));
    }

    [Fact]
    public void ABankSize_KeepsASeededCoreset_TheSameEveryTime()
    {
        WriteImages(20);

        var first = Train(bankSize: 30).Artifacts;
        var again = Train(bankSize: 30).Artifacts;

        Assert.Equal((30, "coreset", 42), (first.Bank.Count, first.Model.Selection, first.Model.Seed));
        Assert.Equal(72L, first.Model.TotalPatches);
        Assert.Equal(first.Bank.Data, again.Bank.Data);
        // ...and it is chosen from the full bank, not other vectors
        var full = Train().Artifacts.Bank;
        var fullRows = Enumerable.Range(0, full.Count).Select(i => string.Join(",", full[i].ToArray())).ToHashSet();
        Assert.All(Enumerable.Range(0, first.Bank.Count), i => Assert.Contains(string.Join(",", first.Bank[i].ToArray()), fullRows));
    }

    [Fact]
    public void ABankSize_FromMorePatchesThanACoresetIsChosenFrom_IsASeededRandomSubsample()
    {
        WriteImages(20);
        var said = new List<string>();

        // a memory allowance of 100 numbers: the 72 patches of 4 numbers are 288
        var first = ImageAnomalyTrainer.Train(_images, "tiny", 30, maxCoresetFloats: 100, said.Add, _pool, default).Artifacts;
        var again = ImageAnomalyTrainer.Train(_images, "tiny", 30, maxCoresetFloats: 100, null, _pool, default).Artifacts;

        Assert.Equal((30, "random", 42), (first.Bank.Count, first.Model.Selection, first.Model.Seed));
        Assert.Equal(first.Bank.Data, again.Bank.Data);
        Assert.Contains(said, line => line.Contains("too many to choose a coreset from", StringComparison.Ordinal));
    }

    [Fact]
    public void ABankSizeAtLeastThePatches_KeepsEveryPatch()
    {
        WriteImages(20);

        var artifacts = Train(bankSize: 10_000).Artifacts;

        Assert.Equal((72, "all"), (artifacts.Bank.Count, artifacts.Model.Selection));
    }

    [Fact]
    public void FewerImagesThanCanBeHeldOut_AreAllKept_WithNoBaseline()
    {
        WriteImages(6);

        var (artifacts, metrics) = Train();

        Assert.Equal((6, 0, 24), (artifacts.Model.Images, artifacts.Model.HeldOutImages, artifacts.Bank.Count));
        Assert.Null(artifacts.Model.Baseline);
        Assert.False(metrics.ContainsKey("normal_score_p95"));
    }

    [Fact]
    public void ANormalImageOfTheBank_ScoresZero_AndAnOddOneDoesNot()
    {
        WriteImages(20);
        var artifacts = Train().Artifacts;
        var embedder = _pool.Get("tiny").Embedder!;

        var seen = embedder.Embed([File.ReadAllBytes(Directory.GetFiles(_images).Order().First())], patches: true).Patches![0].Vectors;
        // a wholly different picture: flat white
        using var white = new SKBitmap(12, 12);
        white.Erase(SKColors.White);
        using var png = SKImage.FromBitmap(white).Encode(SKEncodedImageFormat.Png, 100);
        var odd = embedder.Embed([png.ToArray()], patches: true).Patches![0].Vectors;

        var normalScore = PatchBankScorer.Score(artifacts.Bank, seen).Score;
        var oddScore = PatchBankScorer.Score(artifacts.Bank, odd).Score;

        // (the first image may be the held-out one; either way it is no further than the odd picture)
        Assert.True(oddScore >= normalScore, $"{oddScore} vs {normalScore}");
    }

    [Fact]
    public void ImagesAreListedAtAnyDepth_ByTheExtensionsTheEncoderReads()
    {
        WriteImages(3, Path.Combine(_images, "a", "b"));
        File.WriteAllText(Path.Combine(_images, "notes.txt"), "not an image");
        File.WriteAllBytes(Path.Combine(_images, "x.JPG"), [1, 2, 3]);

        var listed = ImageAnomalyTrainer.ListImages(_images).Select(Path.GetFileName).ToList();

        Assert.Equal(["n000.png", "n001.png", "n002.png", "x.JPG"], listed);
        Assert.Empty(ImageAnomalyTrainer.ListImages(Path.Combine(_root, "nowhere")));
    }

    [Fact]
    public void AFolderWithoutImages_AnImageThatDoesNotDecode_AndAMissingEncoder_AreRefusedByName()
    {
        Assert.Contains("No images", Assert.Throws<InvalidOperationException>(() => Train()).Message);

        WriteImages(12);
        File.WriteAllBytes(Path.Combine(_images, "n005.png"), [1, 2, 3, 4]);
        Assert.Contains("n005.png", Assert.Throws<InvalidOperationException>(() => Train()).Message);

        File.Delete(Path.Combine(_images, "n005.png"));
        Assert.Contains("'other' is not installed",
            Assert.Throws<InvalidOperationException>(() => ImageAnomalyTrainer.Train(_images, "other", null, pool: _pool)).Message);
        Assert.Contains("at least 1", Assert.Throws<InvalidOperationException>(() => Train(bankSize: 0)).Message);
    }


    private string Save(ImageAnomalyArtifacts artifacts)
    {
        var modelPath = Path.Combine(_root, "exp", "model.zip");
        Directory.CreateDirectory(Path.GetDirectoryName(modelPath)!);
        File.WriteAllBytes(modelPath, []);
        artifacts.WriteBeside(modelPath);
        return modelPath;
    }

    [Fact]
    public void ASavedModel_OpensAndScores_EachImageWithAGridOverTheCrop()
    {
        WriteImages(20);
        var modelPath = Save(Train().Artifacts);

        var (scorer, refusal) = ImageAnomalyScorer.Open(modelPath, _pool);
        var scores = scorer!.Score([File.ReadAllBytes(Directory.GetFiles(_images).Order().Last())]);

        Assert.Null(refusal);
        var score = Assert.Single(scores);
        Assert.Equal(4, score.Distances.Length);
        Assert.Equal(score.Distances.Max(), score.Score, 6);
        Assert.True(score.Crop.X1 > score.Crop.X0 && score.Crop.Y1 > score.Crop.Y0);
    }

    [Fact]
    public void ABankBuiltWithOtherWeights_OrNoBank_OrNoRecord_IsRefusedByWhatIsWrong()
    {
        WriteImages(20);
        var artifacts = Train().Artifacts;
        var modelPath = Save(artifacts);

        var other = artifacts.Model with { Encoder = artifacts.Model.Encoder with { Sha256 = "00aa00bb00cc" } };
        other.WriteBeside(modelPath);
        Assert.Contains("other weights", ImageAnomalyScorer.Open(modelPath, _pool).Refusal);

        (artifacts.Model with { Encoder = artifacts.Model.Encoder with { Id = "gone" } }).WriteBeside(modelPath);
        Assert.Contains("encoder 'gone'", ImageAnomalyScorer.Open(modelPath, _pool).Refusal);

        File.Delete(ImageAnomalyModel.BankPathBeside(modelPath));
        artifacts.Model.WriteBeside(modelPath);
        Assert.Contains("patch bank", ImageAnomalyScorer.Open(modelPath, _pool).Refusal);

        File.Delete(ImageAnomalyModel.PathBeside(modelPath));
        Assert.Contains("not an image-anomaly model", ImageAnomalyScorer.Open(modelPath, _pool).Refusal);
        Assert.False(ImageAnomalyScorer.IsImageAnomaly(modelPath));

    }

    [Fact]
    public void ThePatchesAPickKeeps_AreDistinct_Sorted_AndInRange()
    {
        var picked = ImageAnomalyTrainer.PickPatches(1000, 100, new Random(1));

        Assert.Equal(100, picked.Length);
        Assert.Equal(picked.Order().ToArray(), picked);
        Assert.Equal(100, picked.Distinct().Count());
        Assert.All(picked, p => Assert.InRange(p, 0, 999));
        Assert.Equal(ImageAnomalyTrainer.PickPatches(1000, 100, new Random(1)), picked);
        Assert.Equal(Enumerable.Range(0, 5).Select(i => (long)i), ImageAnomalyTrainer.PickPatches(5, 5, new Random(3)));
    }
}
