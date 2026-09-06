using MLoop.Core.Models;
using MLoop.Core.AutoML;
using Xunit;

namespace MLoop.Core.Tests.AutoML;

/// <summary>
/// The registry is process-wide mutable state, so every test that touches it has to be serialized
/// against every other one. xUnit runs distinct test classes in parallel by default, and this
/// assembly does not turn that off, so sharing a collection name is what actually orders them —
/// the isolation cannot live in a convention that each new test is expected to remember.
/// </summary>
[Collection(RegistryCollection)]
public class DeepLearningRegistryTests
{
    internal const string RegistryCollection = "DeepLearningRegistry";

    [Fact]
    public void IsRegistered_false_before_any_registration()
    {
        // Holds only because every test in this collection restores the registry to unregistered
        // when it finishes; the collection is what keeps that true under a parallel runner.
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
                Trainer = TrainerDescriptor.Of("fake"),
                Model = null!,
                Metrics = new Dictionary<string, double>()
            });
        public MLoop.Core.Contracts.IDataProvider? CreateDataLoader(
            string task, Microsoft.ML.MLContext ml, Action<string>? log) => null;
    }
}
