using System.Text.Json;
using MLoop.Search.Tuning;

namespace MLoop.Search.Tests;

/// <summary>
/// The ported tuner against the one MLoop ran before it — ML.NET AutoML 0.23.1's internal <c>CostFrugalTuner</c> —
/// fed the same results, trial by trial: same seed, same space, same proposals to the last bit. «Same algorithm»
/// is not «same proposal order»; only this shows the second. Upstream's clamp of a coordinate above 1
/// (<c>0.99999999</c>) is the one rule the port changes, so the trace runs the port with upstream's clamp.
/// </summary>
/// <remarks>
/// The upstream side is a fixture: AutoML's proposals for three seeds, recorded by driving its tuner through
/// reflection while the package was still referenced (Fixtures/automl-0.23.1-cost-frugal-trace.json) — checked
/// against the authority once, when it was written, and an independent literal since. The losses and costs are
/// computed here from the port's own proposals, which is only the same input as upstream's while the two agree —
/// the first divergence is reported, so a later one cannot hide behind it.
/// <para>
/// An integer setting must match exactly, a real one to a relative 1e-9: the fixture was recorded on one
/// architecture, and the log and power functions the tuner maps through differ in their last bits between arm64 and
/// x64 (measured: 6.580642628446451E-10 against …455E-10). A changed rule moves a proposal far beyond that
/// (<see cref="TheTraceSeesADifferentRule"/>).
/// </para>
/// </remarks>
public sealed class TunerTraceTests
{
    // LightGBM's space as AutoML declares it (CodeGen/lgbm_search_space.json): ranges, log scales, defaults.
    private static readonly (string Name, char Kind, double Min, double Max, bool Log, double Default)[] LightGbm =
    [
        ("NumberOfLeaves", 'i', 4, 32768, false, 4),
        ("MinimumExampleCountPerLeaf", 'i', 20, 1024, true, 20),
        ("LearningRate", 'd', 2e-10, 1, true, 1),
        ("NumberOfTrees", 'i', 4, 32768, false, 4),
        ("SubsampleFraction", 'd', 2e-10, 1, true, 1),
        ("MaximumBinCountPerFeature", 'i', 8, 1024, true, 256),
        ("FeatureFraction", 'd', 2e-10, 1, false, 1),
        ("L1Regularization", 'd', 2e-10, 1, true, 2e-10),
        ("L2Regularization", 'd', 2e-10, 1, true, 1),
    ];

    private static readonly Lazy<Dictionary<string, List<Dictionary<string, double>>>> Upstream = new(() =>
        JsonSerializer.Deserialize<Dictionary<string, List<Dictionary<string, double>>>>(
            File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "automl-0.23.1-cost-frugal-trace.json")))!);

    [Theory]
    [InlineData(0)]
    [InlineData(7)]
    [InlineData(1683)]
    public void ThePortProposesWhatAutoMLProposed_TrialByTrial(int seed)
    {
        var (divergence, distinct) = Trace(seed, stepDimensions: LightGbm.Length, upperClamp: Flow2.UpstreamUpperClamp);

        Assert.True(divergence is null, divergence);
        Assert.True(distinct > 60, $"only {distinct} distinct proposals — the trace did not search");
    }

    [Fact]
    public void TheFixtureHoldsEveryTrialOfEverySeed()
    {
        Assert.Equal(["0", "1683", "7"], Upstream.Value.Keys.Order());
        Assert.All(Upstream.Value.Values, trials =>
        {
            Assert.Equal(120, trials.Count);
            Assert.All(trials, t => Assert.Equal(LightGbm.Select(d => d.Name).Order(), t.Keys.Order()));
        });
    }

    [Fact]
    public void TheTraceSeesADifferentRule()
    {
        // The comparison has to be able to fail: the same port with AutoML's real step scale
        // (one top-level entry) instead of this flat space's nine parts ways within the trace.
        var (divergence, _) = Trace(7, stepDimensions: 1, upperClamp: Flow2.UpstreamUpperClamp);

        Assert.NotNull(divergence);
    }

    private static (string? Divergence, int Distinct) Trace(int seed, int stepDimensions, double upperClamp)
    {
        var space = new SearchSpace(LightGbm.Select(d => d.Kind == 'i'
            ? SearchDimension.Integer(d.Name, (int)d.Min, (int)d.Max, d.Log, (int)d.Default)
            : SearchDimension.Double(d.Name, d.Min, d.Max, d.Log, d.Default)));
        var port = new CostFrugalTuner(space, new TunerRandom(seed), stepDimensions, upperClamp);
        var upstream = Upstream.Value[seed.ToString(System.Globalization.CultureInfo.InvariantCulture)];
        var seen = new HashSet<string>();

        for (var trial = 0; trial < upstream.Count; trial++)
        {
            var mine = port.Propose(trial);
            seen.Add(mine.ToString());
            foreach (var d in LightGbm)
            {
                var expected = upstream[trial][d.Name];
                if (!Agrees(d.Kind, expected, mine[d.Name]))
                    return ($"trial {trial}, {d.Name}: AutoML {expected:R}, port {mine[d.Name]:R}", seen.Count);
            }

            // A deterministic landscape with a basin, and a cost that grows with the model's size.
            var loss = Loss(mine);
            var cost = 10 + mine.Int("NumberOfTrees") * 0.01 + mine.Int("NumberOfLeaves") * 0.001;
            if (trial % 17 == 5)
                loss = double.MaxValue;   // a failed trial, as AutoML scores one
            port.Update(trial, mine, loss, cost);
        }
        return (null, seen.Count);
    }

    private static bool Agrees(char kind, double expected, double actual)
        => kind == 'i' ? expected == actual : Math.Abs(expected - actual) <= 1e-9 * Math.Max(Math.Abs(expected), double.Epsilon);

    private static double Loss(SearchPoint p)
        => Math.Pow(Math.Log(p["LearningRate"]) + 2.3, 2)
           + Math.Pow(Math.Log(p.Int("NumberOfTrees")) - 5, 2) * 0.1
           + Math.Pow(Math.Log(p.Int("NumberOfLeaves")) - 3.5, 2) * 0.1
           + p["L1Regularization"];
}
