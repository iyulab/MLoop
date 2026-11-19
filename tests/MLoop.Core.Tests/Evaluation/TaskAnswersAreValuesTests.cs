using MLoop.Core.Evaluation;

namespace MLoop.Core.Tests.Evaluation;

/// <summary>
/// <see cref="TaskMetadata.AnswersAreValues"/> is the one place that says which tasks answer a number on a scale — the tasks
/// whose feedback can be near the answer but never "the same value". Feedback accuracy reads it (a similarity model
/// with R² 0.94 was read at 0 % and the retraining trigger asked for a new model for good).
/// </summary>
public class TaskAnswersAreValuesTests
{
    [Theory]
    [InlineData("regression")]
    [InlineData("sentence-similarity")]
    [InlineData("forecasting")]
    [InlineData("recommendation")]
    [InlineData("ranking")]
    [InlineData("  Sentence-Similarity ")]
    public void ATaskThatAnswersANumberOnAScaleAnswersAValue(string task) =>
        Assert.True(TaskMetadata.AnswersAreValues(task));

    [Theory]
    [InlineData("binary-classification")]
    [InlineData("multiclass-classification")]
    [InlineData("text-classification")]
    [InlineData("image-classification")]
    [InlineData("ner")]
    [InlineData("question-answering")]
    [InlineData("anomaly-detection")]
    [InlineData("time-series-anomaly")]
    [InlineData("clustering")]
    [InlineData("object-detection")]
    [InlineData("no-such-task")]
    [InlineData("")]
    [InlineData(null)]
    public void AClassATagAnAnswerTextOrAnUnknownTaskDoesNot(string? task) =>
        Assert.False(TaskMetadata.AnswersAreValues(task));

    [Fact]
    public void EveryValueTaskIsATaskTheMetricVocabularyKnows()
    {
        // A value task that no metric table knew would be a typo that silently kept the old reading.
        foreach (var task in new[] { "regression", "sentence-similarity", "forecasting", "recommendation", "ranking" })
            Assert.NotNull(TaskMetadata.PrimaryMetric(task));
    }
}
