using MLoop.Core.Training;

namespace MLoop.Core.Tests.Training;

/// <summary>
/// Which fits spend a time budget. Only the trainer search does; a time-series fit runs its algorithm over
/// the series once and anomaly detection, clustering and ranking fit one pipeline, so auto-time's probe,
/// estimate and "Time Limit" row described time nothing used.
/// The deep-learning case depends on the registered module and is covered with it.
/// </summary>
public class TimeBudgetTests
{
    [Theory]
    [InlineData("forecasting")]
    [InlineData("time-series-anomaly")]
    [InlineData("TimeSeriesAnomaly")]
    public void A_time_series_fit_takes_no_budget(string task) =>
        Assert.NotNull(TrainingRunner.TimeBudgetUnused(task));

    [Fact]
    public void Matrix_factorization_takes_no_budget() =>
        Assert.NotNull(TrainingRunner.TimeBudgetUnused("recommendation"));

    /// <summary>
    /// None of the three reads <c>TimeLimitSeconds</c>; this test once held anomaly detection as a budgeted fit, and
    /// auto-time trained it twice.
    /// </summary>
    [Theory]
    [InlineData("anomaly-detection")]
    [InlineData("clustering")]
    [InlineData("ranking")]
    public void A_single_pipeline_fit_takes_no_budget(string task) =>
        Assert.NotNull(TrainingRunner.TimeBudgetUnused(task));

    [Theory]
    [InlineData("regression")]
    [InlineData("binary-classification")]
    [InlineData("multiclass-classification")]
    public void The_trainer_search_takes_its_budget(string task) =>
        Assert.Null(TrainingRunner.TimeBudgetUnused(task));
}
