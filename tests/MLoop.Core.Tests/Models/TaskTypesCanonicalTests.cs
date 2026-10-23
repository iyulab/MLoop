using MLoop.Core.Models;
using MLoop.Core.Training;

namespace MLoop.Core.Tests.Models;

/// <summary>
/// One answer to "which task is this". Four normalizations were in use — none at all, lowercase,
/// case-insensitive comparison, and lowercase with hyphens and spaces deleted — and the two
/// authorities that already existed disagreed: <c>TaskTypes.IsValid</c> trimmed but did not fold
/// underscores, <c>TrainingRunner.RequiresLabel</c> folded underscores but did not trim.
/// </summary>
public class TaskTypesCanonicalTests
{
    [Fact]
    public void EveryCanonicalNameIsItsOwnCanonicalForm()
    {
        foreach (var task in TaskTypes.All)
            Assert.Equal(task, TaskTypes.Canonical(task));
    }

    [Theory]
    [InlineData("  regression  ")]
    [InlineData("Regression")]
    [InlineData("REGRESSION")]
    [InlineData("\tregression\n")]
    public void ToleratesSpaceAndCase(string spelling)
        => Assert.Equal("regression", TaskTypes.Canonical(spelling));

    [Theory]
    [InlineData("binary_classification")]
    [InlineData("Binary-Classification")]
    [InlineData("BinaryClassification")]
    [InlineData("binary classification")]
    [InlineData(" binary_Classification ")]
    public void FoldsUnderscoresRunTogetherWordsAndSpaces(string spelling)
        => Assert.Equal("binary-classification", TaskTypes.Canonical(spelling));

    /// <summary>
    /// The pre-multi-class spelling. Training, sampling and evaluation already treated it as
    /// binary; only <c>validate</c> rejected it, which is the contradiction this closes.
    /// </summary>
    [Fact]
    public void ClassificationIsTheLegacySpellingOfBinary()
        => Assert.Equal("binary-classification", TaskTypes.Canonical("classification"));

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("regressionn")]
    [InlineData("kmeans")]
    [InlineData("binary")]
    public void UnknownIsNullRatherThanAGuess(string? task)
        => Assert.Null(TaskTypes.Canonical(task));

    /// <summary>An alias is accepted but never advertised.</summary>
    [Fact]
    public void AliasesAreNotPartOfTheAdvertisedVocabulary()
    {
        Assert.DoesNotContain("classification", TaskTypes.All);
        Assert.DoesNotContain("binaryclassification", TaskTypes.All);

        // Item-wise, not substring: the advertised line legitimately contains
        // "binary-classification", which has "classification" inside it.
        var advertised = TaskTypes.Listed.Split(", ");
        Assert.DoesNotContain("classification", advertised);
        Assert.DoesNotContain("binaryclassification", advertised);
        Assert.Equal(TaskTypes.All.Count, advertised.Length);
    }

    [Fact]
    public void IsValidAcceptsExactlyWhatCanonicalResolves()
    {
        foreach (var spelling in new[]
                 {
                     "regression", " Regression ", "binary_classification", "classification",
                     "BinaryClassification", "nonsense", "", "   "
                 })
        {
            Assert.Equal(TaskTypes.Canonical(spelling) != null, TaskTypes.IsValid(spelling));
        }
    }

    // --- the contradictions this closes ---

    /// <summary>
    /// `task: classification` was rejected by validate and accepted as binary by train and
    /// evaluate. Whatever the answer, one answer.
    /// </summary>
    [Fact]
    public void TheLegacySpellingNoLongerSplitsTheProductInTwo()
    {
        Assert.True(TaskTypes.IsValid("classification"));
        Assert.True(TrainingRunner.RequiresLabel("classification"));
        Assert.True(TrainingRunner.SupportsPreFeaturizer("classification"));
    }

    /// <summary>
    /// `task: binary_classification` split the other way: the label and pre-featurizer predicates
    /// folded the underscore, the vocabulary check did not.
    /// </summary>
    [Fact]
    public void TheUnderscoreSpellingNoLongerSplitsTheProductInTwo()
    {
        Assert.True(TaskTypes.IsValid("binary_classification"));
        Assert.True(TrainingRunner.SupportsPreFeaturizer("binary_classification"));
    }

    /// <summary>
    /// A surrounding space used to pass the vocabulary check and fail the predicates, because only
    /// one of the two trimmed.
    /// </summary>
    [Fact]
    public void SurroundingSpaceIsToleratedByBothAuthorities()
    {
        Assert.True(TaskTypes.IsValid(" clustering "));
        Assert.False(TrainingRunner.RequiresLabel(" clustering "));
        Assert.True(TrainingRunner.IsTimeSeriesTask(" forecasting "));
    }

    /// <summary>
    /// The conservative default the label predicate documents: a task it does not recognize still
    /// requires a label, because guessing the other way trains a model on nothing.
    /// </summary>
    [Fact]
    public void AnUnrecognizedTaskStillRequiresALabel()
    {
        Assert.True(TrainingRunner.RequiresLabel("nonsense"));
        Assert.True(TrainingRunner.RequiresLabel(null));
        Assert.False(TrainingRunner.SupportsPreFeaturizer("nonsense"));
        Assert.False(TrainingRunner.IsTimeSeriesTask("nonsense"));
    }
}
