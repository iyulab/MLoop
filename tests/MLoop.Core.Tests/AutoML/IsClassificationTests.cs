using MLoop.Core.AutoML;
using MLoop.Core.Data;
using MLoop.Core.Models;

namespace MLoop.Core.Tests.AutoML;

/// <summary>
/// "Which tasks are classification" had seven inline answers because two questions were sharing
/// one name: whether a model predicts a class, and whether its classes can be counted in a CSV
/// column. The second is the first composed with <see cref="DataLoaderFactory.IsDirectoryBased"/>,
/// which is why there is one predicate here and not two.
/// </summary>
public class IsClassificationTests
{
    [Theory]
    [InlineData("binary-classification")]
    [InlineData("multiclass-classification")]
    [InlineData("text-classification")]
    [InlineData("image-classification")]
    public void PredictsAClass(string task)
        => Assert.True(AutoMLRunner.IsClassification(task));

    [Theory]
    [InlineData("regression")]
    [InlineData("clustering")]
    [InlineData("anomaly-detection")]
    [InlineData("ranking")]
    [InlineData("forecasting")]
    [InlineData("time-series-anomaly")]
    [InlineData("recommendation")]
    [InlineData("object-detection")]
    [InlineData("ner")]
    [InlineData("question-answering")]
    [InlineData("sentence-similarity")]
    public void DoesNot(string task)
        => Assert.False(AutoMLRunner.IsClassification(task));

    /// <summary>
    /// The copy this replaced compared ordinally against unfolded literals, so a task string that
    /// had not been through the config merge answered "not a classification task" and its
    /// predictions were read as a regression score.
    /// </summary>
    [Theory]
    [InlineData("Binary-Classification")]
    [InlineData("binary_classification")]
    [InlineData(" classification ")]
    [InlineData("BinaryClassification")]
    public void ToleratesEverySpellingTheVocabularyDoes(string spelling)
        => Assert.True(AutoMLRunner.IsClassification(spelling));

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("nonsense")]
    public void UnknownIsNotClassification(string? task)
        => Assert.False(AutoMLRunner.IsClassification(task));

    // --- the composed question ---

    private static bool CountsClassesInAColumn(string task)
        => AutoMLRunner.IsClassification(task) && !DataLoaderFactory.IsDirectoryBased(task);

    /// <summary>
    /// <c>text-classification</c> reads a CSV with a label column exactly like the other two, and
    /// had been excluded from every site that asks this — so a text classifier got an unstratified
    /// split, a class count of zero in its time estimate, and no per-class minimum check.
    /// </summary>
    [Theory]
    [InlineData("binary-classification")]
    [InlineData("multiclass-classification")]
    [InlineData("text-classification")]
    public void ClassesLiveInAColumnForCsvBackedClassification(string task)
        => Assert.True(CountsClassesInAColumn(task));

    /// <summary>Labels that are directory names are not a column to count.</summary>
    [Fact]
    public void ImageClassificationPredictsAClassButHasNoColumnToCount()
    {
        Assert.True(AutoMLRunner.IsClassification("image-classification"));
        Assert.False(CountsClassesInAColumn("image-classification"));
    }

    /// <summary>
    /// The two questions are genuinely different — if they ever coincide, one of them has stopped
    /// being asked and the composition at the call sites is dead weight.
    /// </summary>
    [Fact]
    public void TheTwoQuestionsDifferOnAtLeastOneTask()
    {
        var differ = TaskTypes.All
            .Where(t => AutoMLRunner.IsClassification(t) != CountsClassesInAColumn(t))
            .ToList();

        Assert.NotEmpty(differ);
        Assert.Contains("image-classification", differ);
    }
}
