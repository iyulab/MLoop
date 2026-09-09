using MLoop.CLI.Infrastructure.Display;
using MLoop.Core.Models;

namespace MLoop.Tests.Commands;

/// <summary>
/// The Trainer column shows a trainer, and nothing it leaves out is lost.
/// </summary>
/// <remarks>
/// What AutoML records is the whole pipeline it assembled, and a column headed "Trainer" printing
/// <c>ReplaceMissingValues=&gt;Concatenate=&gt;FastTreeBinary</c> wrapped across five lines in an
/// eighty-column terminal — measured, with no fallback annotation involved at all, which is wider
/// than the problem had been recorded as.
/// </remarks>
public class TrainerDisplayTests
{
    [Theory]
    [InlineData("ReplaceMissingValues=>Concatenate=>FastTreeBinary", "FastTreeBinary")]
    [InlineData("Concatenate=>LightGbmBinary", "LightGbmBinary")]
    [InlineData("LbfgsLogisticRegression", "LbfgsLogisticRegression")]
    [InlineData("A=>B=> C ", "C")]
    public void TheColumnShowsTheTrainerAndNotTheTransformsBeforeIt(string full, string expected)
    {
        Assert.Equal(expected, TrainerDisplay.Short(full));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void AMissingNameRendersAsAnAbsence(string? full)
    {
        Assert.Equal("-", TrainerDisplay.Short(full));
    }

    [Fact]
    public void ShorteningKeepsWhatTellsTwoRunsApart()
    {
        // A leaderboard compares configurations of one trainer. Dropping the parameters would make
        // `KMeans (k=2)` and `KMeans (k=3)` render identically, and the ranking would stop meaning
        // anything. Only the chain prefix is the same on every row, so only it can go.
        var trainer = new TrainerDescriptor
        {
            Name = "Concatenate=>KMeans",
            Params = new Dictionary<string, string> { ["k"] = "3" },
        };

        Assert.Equal("KMeans (k=3)", TrainerDisplay.Short(trainer));
    }

    [Fact]
    public void AFallbackReasonIsNotDraggedIntoTheColumn()
    {
        // It is a sentence; the column is a column. It belongs under the table.
        var trainer = new TrainerDescriptor
        {
            Name = "SdcaLogisticRegression",
            FallbackReason = "manual fallback: AutoML AUC failure",
        };

        Assert.Equal("SdcaLogisticRegression", TrainerDisplay.Short(trainer));
    }

    [Fact]
    public void WhatTheColumnLeftOutIsPrintedUnderTheTable()
    {
        var note = TrainerDisplay.Footnote(
            "[exp-001]",
            new TrainerDescriptor { Name = "Concatenate=>FastTreeBinary" });

        Assert.Equal("[exp-001] Concatenate=>FastTreeBinary", note);
    }

    [Fact]
    public void TheFallbackReasonComesFromTheFieldNotFromTheDisplayString()
    {
        // Recovering it by splitting `Display` is what the descriptor's own documentation forbids:
        // a change of wording would become a change of behaviour.
        var note = TrainerDisplay.Footnote(
            "[exp-002]",
            new TrainerDescriptor { Name = "SdcaLogisticRegression", FallbackReason = "manual fallback: AutoML AUC failure" });

        Assert.Equal("[exp-002] manual fallback: AutoML AUC failure", note);
    }

    [Fact]
    public void ARowWithNothingToAddGetsNoFootnote()
    {
        Assert.Null(TrainerDisplay.Footnote("[exp-003]", new TrainerDescriptor { Name = "FastForestBinary" }));
        Assert.Null(TrainerDisplay.Footnote("[exp-004]", null));
    }

    [Fact]
    public void BothHalvesAppearWhenBothApply()
    {
        var note = TrainerDisplay.Footnote(
            "[exp-005]",
            new TrainerDescriptor { Name = "Concatenate=>SdcaLogisticRegression", FallbackReason = "manual fallback: AutoML AUC failure" });

        Assert.Equal(
            "[exp-005] Concatenate=>SdcaLogisticRegression — manual fallback: AutoML AUC failure",
            note);
    }
}
