using MLoop.Core.AutoML;

namespace MLoop.Core.Tests.AutoML;

public class AutoTimeIntegrationTests
{
    [Fact]
    public void FullPipeline_StaticThenReactive()
    {
        int rows = 5000, cols = 10, classes = 4;
        var task = "multiclass-classification";

        var staticEst = TimeEstimator.EstimateStatic(rows, cols, task, classes);
        Assert.InRange(staticEst, 30, 1800);

        var probeTime = TimeEstimator.GetProbeTime(staticEst);
        Assert.True(probeTime <= 30);
        Assert.True(probeTime >= 10);

        var probe = new ProbeResult
        {
            BestMetric = 0.75,
            ProbeTimeSeconds = probeTime,
            TrialsCompleted = 3
        };

        var finalTime = TimeEstimator.EstimateReactive(probe, staticEst);
        Assert.True(finalTime >= staticEst, "Moderate probe should increase time");
    }

    [Fact]
    public void FullPipeline_ConvergesEarly()
    {
        int rows = 1000, cols = 3, classes = 3;
        var task = "multiclass-classification";

        var staticEst = TimeEstimator.EstimateStatic(rows, cols, task, classes);
        var probeTime = TimeEstimator.GetProbeTime(staticEst);

        var probe = new ProbeResult
        {
            BestMetric = 0.98,
            ProbeTimeSeconds = probeTime,
            TrialsCompleted = 5
        };

        var finalTime = TimeEstimator.EstimateReactive(probe, staticEst);
        Assert.True(finalTime <= staticEst, "High-metric probe should reduce time");
    }
}
