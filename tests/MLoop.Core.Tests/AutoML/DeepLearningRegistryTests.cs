using MLoop.Core.AutoML;
using Xunit;

namespace MLoop.Core.Tests.AutoML;

public class DeepLearningRegistryTests
{
    [Fact]
    public void IsRegistered_false_before_any_registration()
    {
        // 주의: static 상태이므로 이 테스트는 Register가 호출되지 않은 프로세스에서만 유효.
        // Register를 호출하는 테스트와 같은 어셈블리에 두지 않는다(Core.Tests는 미등록 유지).
        Assert.False(DeepLearningRegistry.IsRegistered);
    }

    [Fact]
    public void Register_sets_Current()
    {
        var module = new FakeModule();
        DeepLearningRegistry.Register(module);
        Assert.Same(module, DeepLearningRegistry.Current);
        DeepLearningRegistry.Register(null!); // 정리(테스트 격리)
    }

    private sealed class FakeModule : IDeepLearningModule
    {
        public bool CanHandleTask(string task) => task == "ner";
        public Task<AutoMLResult> TrainAsync(Microsoft.ML.MLContext ml, Action<string> log,
            string task, Microsoft.ML.IDataView train, Microsoft.ML.IDataView test,
            MLoop.Core.Models.TrainingConfig config,
            IProgress<MLoop.Core.Models.TrainingProgress>? progress, CancellationToken ct)
            => Task.FromResult(new AutoMLResult
            {
                BestTrainer = "fake",
                Model = null!,
                Metrics = new Dictionary<string, double>()
            });
        public MLoop.Core.Contracts.IDataProvider? CreateDataLoader(
            string task, Microsoft.ML.MLContext ml, Action<string>? log) => null;
    }
}
