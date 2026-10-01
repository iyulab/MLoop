using MLoop.Core.AutoML;

namespace MLoop.Core.Tests.AutoML;

/// <summary>
/// Which fits spend a time budget. Only an AutoML search does; a time-series fit runs its algorithm over
/// the series once, so auto-time's probe, estimate and "Time Limit" row described time nothing used.
/// The deep-learning case depends on the registered module and is covered with it.
/// </summary>
public class TimeBudgetTests
{
    [Theory]
    [InlineData("forecasting")]
    [InlineData("time-series-anomaly")]
    [InlineData("TimeSeriesAnomaly")]
    public void A_time_series_fit_takes_no_budget(string task) =>
        Assert.NotNull(AutoMLRunner.TimeBudgetUnused(task));

    [Theory]
    [InlineData("regression")]
    [InlineData("binary-classification")]
    [InlineData("anomaly-detection")]
    public void An_automl_search_takes_its_budget(string task) =>
        Assert.Null(AutoMLRunner.TimeBudgetUnused(task));
}
