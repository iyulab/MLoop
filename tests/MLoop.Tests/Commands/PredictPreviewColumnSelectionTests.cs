using MLoop.CLI.Commands;

namespace MLoop.Tests.Commands;

/// <summary>
/// The preview after <c>mloop predict</c> must show what was predicted. A prediction file appends
/// its answer columns to the row it was given, so a table that renders every column spends the
/// terminal's width on the user's own input and leaves the answer a single character wide.
/// </summary>
/// <remarks>
/// Measured on this repository's churn example at 80 columns, before this selection existed: 24
/// columns, every header elided to an ellipsis, values running vertically one letter per line, and
/// <c>PredictedLabel</c>/<c>Score</c>/<c>Probability</c>/<c>Confidence</c> rendered blank. The exit
/// code was 0 and no assertion looked at it — which is why this is pinned here rather than left to
/// the release walkthrough that found it.
/// </remarks>
public class PredictPreviewColumnSelectionTests
{
    /// <summary>The churn example's real shape: 20 input columns, four appended by the run.</summary>
    private static readonly string[] ChurnOutputHeaders =
    [
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges",
        "PredictedLabel", "Score", "Probability", "Confidence",
    ];

    private static readonly string[] ChurnInputHeaders = ChurnOutputHeaders[..20];

    [Fact]
    public void EveryAnswerColumnSurvivesTheNarrowestTerminalThisRenders()
    {
        var (shown, omitted) = PredictCommand.SelectPreviewColumns(ChurnOutputHeaders, ChurnInputHeaders, 80);

        Assert.Contains("PredictedLabel", shown);
        Assert.Contains("Score", shown);
        Assert.Contains("Probability", shown);
        Assert.Contains("Confidence", shown);
        Assert.True(omitted > 0, "The churn example has more input columns than 80 characters can carry.");
        Assert.Equal(ChurnOutputHeaders.Length, shown.Length + omitted);
    }

    [Fact]
    public void AtLeastOneInputColumnStaysSoARowCanBeToldFromTheNext()
    {
        var (shown, _) = PredictCommand.SelectPreviewColumns(ChurnOutputHeaders, ChurnInputHeaders, 80);

        Assert.Contains("customerID", shown);
    }

    [Fact]
    public void ColumnsKeepTheOrderTheFileWroteThem()
    {
        var (shown, _) = PredictCommand.SelectPreviewColumns(ChurnOutputHeaders, ChurnInputHeaders, 80);

        var positions = shown.Select(h => Array.IndexOf(ChurnOutputHeaders, h)).ToArray();
        Assert.Equal(positions.OrderBy(p => p), positions);
    }

    [Fact]
    public void AWiderTerminalKeepsMoreOfTheInput()
    {
        var (narrow, _) = PredictCommand.SelectPreviewColumns(ChurnOutputHeaders, ChurnInputHeaders, 80);
        var (wide, _) = PredictCommand.SelectPreviewColumns(ChurnOutputHeaders, ChurnInputHeaders, 200);

        Assert.True(wide.Length > narrow.Length,
            $"Expected a 200-column terminal to show more than {narrow.Length}, it showed {wide.Length}.");
    }

    [Fact]
    public void AFileWithoutTheOriginalFeaturesLosesNothing()
    {
        // predict --no-features: the output is answers only, so there is nothing to leave out and
        // the preview must not start trimming what it was asked to show.
        string[] answersOnly = ["PredictedLabel", "Score", "Probability", "Confidence"];

        var (shown, omitted) = PredictCommand.SelectPreviewColumns(answersOnly, ChurnInputHeaders, 80);

        Assert.Equal(answersOnly, shown);
        Assert.Equal(0, omitted);
    }

    [Fact]
    public void AnUnreadableInputHeaderShowsEveryColumnRatherThanNone()
    {
        // The header of the input file is what tells an answer from an echo. With none, the safe
        // reading is that every column is an answer — the behaviour before this selection existed —
        // rather than deciding the whole row is context and showing nothing.
        var (shown, omitted) = PredictCommand.SelectPreviewColumns(ChurnOutputHeaders, [], 80);

        Assert.Equal(ChurnOutputHeaders, shown);
        Assert.Equal(0, omitted);
    }

    [Fact]
    public void AByteOrderMarkOnTheFirstColumnDoesNotMakeItLookPredicted()
    {
        // The prediction file is written with a BOM and the input file here is not. Compared raw,
        // "﻿customerID" is a name the input never had — so the identifier column would be
        // classified as something the model produced, and the very column this keeps for context
        // would instead be kept as an answer.
        string[] withBom = ["﻿customerID", "tenure", "PredictedLabel", "Score"];

        var (shown, _) = PredictCommand.SelectPreviewColumns(withBom, ["customerID", "tenure"], 80);

        Assert.Equal(withBom, shown); // narrow enough that everything fits either way
        var (shownNarrow, _) = PredictCommand.SelectPreviewColumns(
            [.. withBom, "Probability", "Confidence", "Extra1", "Extra2", "Extra3"],
            ["customerID", "tenure", "Extra1", "Extra2", "Extra3"],
            28);

        Assert.DoesNotContain("Extra3", shownNarrow);
        Assert.Contains("﻿customerID", shownNarrow);
        Assert.Contains("PredictedLabel", shownNarrow);
    }
}
