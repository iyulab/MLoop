using System.Reflection;
using Microsoft.ML.AutoML;
using MLoop.Search.Tuning;
using SearchSpace = MLoop.Search.Tuning.SearchSpace;
using UniformDoubleOption = Microsoft.ML.SearchSpace.Option.UniformDoubleOption;
using UniformIntOption = Microsoft.ML.SearchSpace.Option.UniformIntOption;
using Upstream = Microsoft.ML.SearchSpace;

namespace MLoop.Core.Tests.Search;

/// <summary>
/// The ported tuner against the one MLoop ran until now — ML.NET AutoML 0.23's internal
/// <c>CostFrugalTuner</c>, reached by reflection — fed the same results, trial by trial: same seed, same
/// space, same proposals to the last bit. «Same algorithm» is not «same proposal order»; only this shows
/// the second. Upstream's clamp of a coordinate above 1 (<c>0.99999999</c>) is the
/// one rule the port changes, so the trace runs the port with upstream's clamp.
/// </summary>
/// <remarks>
/// TODO(AutoML removal): when the AutoML package goes, freeze one trace as a fixture — a literal checked once
/// against the authority (CLAUDE.md testing rules).
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
    public void TheTraceSeesADifferentRule()
    {
        // The comparison has to be able to fail: the same port with AutoML's real step scale
        // (one top-level entry) instead of this flat space's nine parts ways within the trace.
        var (divergence, _) = Trace(7, stepDimensions: 1, upperClamp: Flow2.UpstreamUpperClamp);

        Assert.NotNull(divergence);
    }

    private static (string? Divergence, int Distinct) Trace(int seed, int stepDimensions, double upperClamp)
    {
        var ours = new SearchSpace(LightGbm.Select(d => d.Kind == 'i'
            ? SearchDimension.Integer(d.Name, (int)d.Min, (int)d.Max, d.Log, (int)d.Default)
            : SearchDimension.Double(d.Name, d.Min, d.Max, d.Log, d.Default)));
        var theirsSpace = new Upstream.SearchSpace();
        foreach (var d in LightGbm)
        {
            theirsSpace[d.Name] = d.Kind == 'i'
                ? new UniformIntOption((int)d.Min, (int)d.Max, d.Log, (int)d.Default)
                : new UniformDoubleOption(d.Min, d.Max, d.Log, d.Default);
        }

        var port = new CostFrugalTuner(ours, new TunerRandom(seed), stepDimensions, upperClamp);
        var upstream = UpstreamTuner.Create(theirsSpace, seed);
        var seen = new HashSet<string>();

        for (var trial = 0; trial < 120; trial++)
        {
            var mine = port.Propose(trial);
            var theirs = upstream.Propose(trial);
            seen.Add(mine.ToString());
            foreach (var d in LightGbm)
            {
                var expected = d.Kind == 'i' ? theirs[d.Name].AsType<int>() : theirs[d.Name].AsType<double>();
                if (expected != mine[d.Name])
                    return ($"trial {trial}, {d.Name}: AutoML {expected:R}, port {mine[d.Name]:R}", seen.Count);
            }

            // A deterministic landscape with a basin, and a cost that grows with the model's size.
            var loss = Loss(mine);
            var cost = 10 + mine.Int("NumberOfTrees") * 0.01 + mine.Int("NumberOfLeaves") * 0.001;
            if (trial % 17 == 5)
                loss = double.MaxValue;   // a failed trial, as AutoML scores one
            port.Update(trial, mine, loss, cost);
            upstream.Update(trial, theirs, loss, cost);
        }
        return (null, seen.Count);
    }

    private static double Loss(SearchPoint p)
        => Math.Pow(Math.Log(p["LearningRate"]) + 2.3, 2)
           + Math.Pow(Math.Log(p.Int("NumberOfTrees")) - 5, 2) * 0.1
           + Math.Pow(Math.Log(p.Int("NumberOfLeaves")) - 3.5, 2) * 0.1
           + p["L1Regularization"];

    /// <summary>AutoML's internal tuner, driven through its public trial types.</summary>
    private sealed class UpstreamTuner
    {
        private readonly object _tuner;
        private readonly MethodInfo _propose;
        private readonly MethodInfo _update;

        private UpstreamTuner(object tuner)
        {
            _tuner = tuner;
            _propose = tuner.GetType().GetMethod("Propose", [typeof(TrialSettings)])!;
            _update = tuner.GetType().GetMethod("Update", [typeof(TrialResult)])!;
        }

        public static UpstreamTuner Create(Upstream.SearchSpace space, int seed)
        {
            var type = typeof(AutoMLExperiment).Assembly.GetType("Microsoft.ML.AutoML.CostFrugalTuner", throwOnError: true)!;
            var ctor = type.GetConstructor(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
                [typeof(Upstream.SearchSpace), typeof(Upstream.Parameter), typeof(IEnumerable<TrialResult>), typeof(int?)])
                ?? throw new InvalidOperationException("AutoML's CostFrugalTuner constructor changed shape.");
            return new UpstreamTuner(ctor.Invoke([space, space.SampleFromFeatureSpace(space.Default), null, seed]));
        }

        public Upstream.Parameter Propose(int trialId)
            => (Upstream.Parameter)_propose.Invoke(_tuner, [new TrialSettings { TrialId = trialId }])!;

        public void Update(int trialId, Upstream.Parameter parameter, double loss, double cost)
            => _update.Invoke(_tuner, [new TrialResult
            {
                TrialSettings = new TrialSettings { TrialId = trialId, Parameter = parameter },
                Loss = loss,
                DurationInMilliseconds = cost,
            }]);
    }
}
