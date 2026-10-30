using System.Globalization;
using FluentAssertions;
using MLoop.Core.Monitoring;
using MLoop.Ops.Services;

namespace MLoop.Ops.Tests;

// The retraining trigger reads the PSI against what chance alone gives, not against the bare threshold: a training set of 150 rows makes the
// PSI of a window with nothing changed exceed 0.25 in a third of the windows at fifty columns (measured in a simulation), and the bare threshold
// asked for a retraining on that.
public class InputMovedTests
{
    private const int Columns = 50;

    private static string?[] Row(Random rng, double shift)
    {
        var row = new string?[Columns];
        for (var c = 0; c < Columns; c++)
        {
            if (c % 3 == 2)
            {
                var u = rng.NextDouble();
                row[c] = u < 0.4 ? "a" : u < 0.7 ? "b" : u < 0.85 ? "c" : u < 0.95 ? "d" : "e";
            }
            else
            {
                var gauss = Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
                row[c] = rng.NextDouble() < 0.02 ? null : (gauss + (c == 0 ? shift : 0)).ToString("R", CultureInfo.InvariantCulture);
            }
        }
        return row;
    }

    private static (DriftProfile Profile, Random Rng) ProfileOf(int trainRows, int seed)
    {
        var rng = new Random(seed);
        var spec = Enumerable.Range(0, Columns).Select(c => ($"c{c}", c % 3 != 2)).ToList();
        return (DriftProfile.Build(Enumerable.Range(0, trainRows).Select(_ => Row(rng, 0)).ToList(), spec), rng);
    }

    private static DriftReport Compare(DriftProfile profile, Random rng, double shift)
    {
        var rows = Enumerable.Range(0, 300).Select(_ =>
        {
            var r = Row(rng, shift);
            return (IReadOnlyDictionary<string, object?>)Enumerable.Range(0, Columns).ToDictionary(c => $"c{c}", c => (object?)r[c]);
        }).ToList();
        return DriftMeasure.Compare(profile, rows);
    }

    [Fact]
    public void AnUnchangedWindow_AgainstASmallTrainingSet_DoesNotAskForARetraining()
    {
        var (profile, rng) = ProfileOf(trainRows: 150, seed: 31);
        var reports = Enumerable.Range(0, 100).Select(_ => Compare(profile, rng, 0)).ToList();

        var bare = reports.Count(r => r.MaxPsi >= 0.25);
        var read = reports.Count(r => CompositeRetrainingTrigger.InputMoved(r, 0.25));

        // the bare threshold is what the trigger used to apply
        bare.Should().BeGreaterThan(20, "the premise: a training set this small makes chance alone cross 0.25");
        // 15 of 200 in the simulation, 16 of 100 here: one 150-row profile is one draw, and its own noise moves the rate between about 8 % and 16 %
        // — above the 5 % the bound is cut for. What this pins is "well under the bare reading", not the family error rate.
        read.Should().BeLessThanOrEqualTo(25);
        read.Should().BeLessThan(bare / 2);
    }

    [Fact]
    public void AnUnchangedWindow_AgainstALargeTrainingSet_NeverAsksForARetraining()
    {
        var (profile, rng) = ProfileOf(trainRows: 5_000, seed: 32);

        Enumerable.Range(0, 60).Count(_ => CompositeRetrainingTrigger.InputMoved(Compare(profile, rng, 0), 0.25)).Should().Be(0);
    }

    [Theory]
    [InlineData(150)]
    [InlineData(5_000)]
    public void AColumnMovedByASigma_IsCaught_WhateverTheTrainingSize(int trainRows)
    {
        // the guard can see: a move this size reads over the threshold and over what chance gives
        var (profile, rng) = ProfileOf(trainRows, seed: 33);

        Enumerable.Range(0, 30).Count(_ => CompositeRetrainingTrigger.InputMoved(Compare(profile, rng, 1.0), 0.25)).Should().Be(30);
    }
}
