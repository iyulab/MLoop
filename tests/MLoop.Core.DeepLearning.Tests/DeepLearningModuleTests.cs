using MLoop.Core.DeepLearning;
using Xunit;

namespace MLoop.Core.DeepLearning.Tests;

public class DeepLearningModuleTests
{
    [Theory]
    [InlineData("object-detection", true)]
    [InlineData("ner", true)]
    [InlineData("image-classification", true)]
    [InlineData("regression", false)]
    [InlineData("binary-classification", false)]
    public void CanHandleTask_matches_dl_tasks(string task, bool expected)
        => Assert.Equal(expected, new DeepLearningModule().CanHandleTask(task));

    [Fact]
    public void CreateDataLoader_returns_null_for_tabular()
        => Assert.Null(new DeepLearningModule().CreateDataLoader(
            "regression", new Microsoft.ML.MLContext(), null));
}
