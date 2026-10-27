using System.Diagnostics;
using MLoop.Core.Embedding;
using MLoop.Core.ImageAnomaly;
using MLoop.Core.Storage;

namespace MLoop.Core.Tests.ImageAnomaly;

public class PatchBankTests(ITestOutputHelper output) : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("mloop-bank-").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* temp */ }
    }

    private static float[] Unit(Random random, int dimension)
    {
        var v = Enumerable.Range(0, dimension).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        var norm = MathF.Sqrt(v.Sum(x => x * x));
        return v.Select(x => x / norm).ToArray();
    }

    private static List<float[]> Units(int count, int dimension, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, count).Select(_ => Unit(random, dimension)).ToList();
    }

    [Theory]
    [InlineData(1000, 384)]   // not a multiple of the 256-row block
    [InlineData(256, 384)]    // exactly one block
    [InlineData(300, 37)]     // a dimension no vector width divides
    [InlineData(5, 3)]        // narrower than a vector
    public void TheNearestDistances_AreThoseOfAPlainScan(int bankSize, int dimension)
    {
        var vectors = Units(bankSize, dimension, seed: 1);
        var queries = Units(20, dimension, seed: 2);
        var bank = PatchBank.Of(vectors, dimension);

        var fast = PatchBankScorer.NearestDistances(bank, queries);
        var plain = queries.Select(q => (float)(1 - vectors.Max(b => b.Zip(q, (x, y) => (double)x * y).Sum()))).Select(d => Math.Max(0f, d)).ToArray();

        Assert.Equal(plain.Length, fast.Length);
        for (var i = 0; i < fast.Length; i++)
            Assert.Equal(plain[i], fast[i], 4);
    }

    [Fact]
    public void AQueryThatIsInTheBank_IsAtDistanceZero_AndTheScoreIsTheLargestDistance()
    {
        var vectors = Units(500, 64, seed: 3);
        var bank = PatchBank.Of(vectors, 64);
        var stranger = Unit(new Random(99), 64);

        var (score, distances) = PatchBankScorer.Score(bank, [vectors[7], vectors[400], stranger]);

        Assert.Equal(0f, distances[0], 4);
        Assert.Equal(0f, distances[1], 4);
        Assert.True(distances[2] > 0.05f);
        Assert.Equal(distances[2], score);
    }

    [Fact]
    public void AnEmptyBank_OrAWrongLength_IsRefused()
    {
        var empty = PatchBank.Of([], 8);
        Assert.Throws<ArgumentException>(() => PatchBankScorer.NearestDistances(empty, [new float[8]]));
        var bank = PatchBank.Of(Units(4, 8, 1), 8);
        Assert.Throws<ArgumentException>(() => PatchBankScorer.NearestDistances(bank, [new float[7]]));
        Assert.Throws<ArgumentException>(() => PatchBank.Of([new float[3]], 8));
    }

    [Fact]
    public void ABank_RoundTripsThroughItsFile()
    {
        var bank = PatchBank.Of(Units(300, 48, 5), 48);
        var path = Path.Combine(_dir, "patch-bank.bin");

        bank.Write(path);
        var read = PatchBank.Read(path);

        Assert.Equal((300, 48), (read.Count, read.Dimension));
        Assert.Equal(bank.Data, read.Data);
    }

    [Fact]
    public void AFileThatIsNotABank_OrIsCutShort_IsRefusedByName()
    {
        var path = Path.Combine(_dir, "x.bin");
        File.WriteAllText(path, "not a bank at all, just text");
        Assert.Contains("not a patch bank", Assert.Throws<InvalidDataException>(() => PatchBank.Read(path)).Message);

        var good = Path.Combine(_dir, "good.bin");
        PatchBank.Of(Units(10, 8, 1), 8).Write(good);
        var bytes = File.ReadAllBytes(good);
        File.WriteAllBytes(path, bytes[..^10]);
        Assert.Contains("bytes of vectors", Assert.Throws<InvalidDataException>(() => PatchBank.Read(path)).Message);
    }

    [Fact]
    public void TheBaseline_IsTheNearestRankPercentilesOfTheHeldOutScores()
    {
        var baseline = ScoreBaseline.Of(Enumerable.Range(1, 100).Select(i => (double)i).ToList())!;

        Assert.Equal((100, 50.0, 95.0, 99.0, 100.0), (baseline.Images, baseline.Median, baseline.P95, baseline.P99, baseline.Max));
        Assert.Null(ScoreBaseline.Of([]));
    }

    private static EncoderIdentity Identity(string id = "dinov2-small", string sha = "f22797ea1234", string pre = "pillow/2", int dim = 384)
        => new(id, "1", sha, dim, Normalized: true, pre);

    [Fact]
    public void TheRecord_RoundTripsBesideTheModel_AndAnotherVectorSpaceIsNamed()
    {
        var model = Path.Combine(_dir, ExperimentLayout.ModelFileName);
        Assert.Null(ImageAnomalyModel.ReadBeside(model));
        var record = new ImageAnomalyModel(Identity(), new PatchGrid(16, 16, 14, 224), Images: 90, HeldOutImages: 10,
            TotalPatches: 25_600, BankPatches: 23_040, "all", Seed: null, ScoreBaseline.Of([0.1, 0.2, 0.3]));

        record.WriteBeside(model);
        var read = ImageAnomalyModel.ReadBeside(model)!;

        Assert.Equal(record.Encoder, read.Encoder);
        Assert.Equal(record.Grid, read.Grid);
        Assert.Equal(record.Baseline, read.Baseline);
        Assert.Null(read.Mismatch(Identity()));
        Assert.Contains("other weights", read.Mismatch(Identity(sha: "00aa00bb"))!);
        Assert.Contains("pillow/3", read.Mismatch(Identity(pre: "pillow/3"))!);
        Assert.Contains("'other'", read.Mismatch(Identity(id: "other"))!);
        Assert.Contains("768", read.Mismatch(Identity(dim: 768))!);
    }

    // Reported, not asserted: the cost of one image's 256 patches against a bank of 600 images' patches, on this machine.
    [Fact]
    [Trait("Category", "Slow")]
    public void TheCostOfOneImageAgainstALargeBank_IsReported()
    {
        var random = new Random(7);
        var data = new float[153_600 * 384];
        for (var i = 0; i < data.Length; i++) data[i] = (float)random.NextDouble() - 0.5f;
        var bank = PatchBank.FromRows(data, 384);
        var queries = Units(256, 384, 8);

        PatchBankScorer.NearestDistances(bank, queries);   // warm up
        var watch = Stopwatch.StartNew();
        for (var run = 0; run < 3; run++) PatchBankScorer.NearestDistances(bank, queries);
        watch.Stop();

        output.WriteLine($"BENCH 153,600 x 384 bank, 256 queries: {watch.ElapsedMilliseconds / 3.0:F0} ms per image ({Environment.ProcessorCount} logical cores)");
        Assert.True(watch.ElapsedTicks > 0);
    }
}
