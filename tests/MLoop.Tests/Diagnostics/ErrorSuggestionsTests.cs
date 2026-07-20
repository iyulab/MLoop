using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.Core.AutoML;

namespace MLoop.Tests.Diagnostics;

public class ErrorSuggestionsTests
{
    [Fact]
    public void GetSuggestions_FileNotFound_SuggestsCheckPath()
    {
        var ex = new FileNotFoundException("Could not find file 'test.csv'");
        var suggestions = ErrorSuggestions.GetSuggestions(ex);

        Assert.NotEmpty(suggestions);
        Assert.Contains(suggestions, s => s.Contains("file path"));
    }

    [Fact]
    public void GetSuggestions_FileNotFoundInTraining_SuggestsDataFile()
    {
        var ex = new FileNotFoundException("File not found");
        var suggestions = ErrorSuggestions.GetSuggestions(ex, "training");

        Assert.Contains(suggestions, s => s.Contains("datasets/train.csv"));
    }

    [Fact]
    public void GetSuggestions_SchemaError_SuggestsColumnCheck()
    {
        var ex = new Exception("Schema mismatch: column 'price' not found");
        var suggestions = ErrorSuggestions.GetSuggestions(ex);

        Assert.Contains(suggestions, s => s.Contains("columns"));
    }

    [Fact]
    public void GetSuggestions_LabelNotFound_SuggestsLabelUpdate()
    {
        var ex = new Exception("Label column not found in data");
        var suggestions = ErrorSuggestions.GetSuggestions(ex);

        Assert.Contains(suggestions, s => s.Contains("label"));
    }

    [Fact]
    public void GetSuggestions_PromoteContext_SuggestsList()
    {
        var ex = new FileNotFoundException("Experiment not found: default/exp-999");
        var suggestions = ErrorSuggestions.GetSuggestions(ex, "promote");

        Assert.Contains(suggestions, s => s.Contains("mloop list"));
    }

    [Fact]
    public void GetSuggestions_PreprocessingContext_SuggestsDryRun()
    {
        var ex = new InvalidOperationException("Unknown prep step type");
        var suggestions = ErrorSuggestions.GetSuggestions(ex, "preprocessing");

        Assert.Contains(suggestions, s => s.Contains("dry-run"));
    }

    [Fact]
    public void GetSuggestions_ConfigError_SuggestsYamlCheck()
    {
        var ex = new Exception("Invalid mloop.yaml configuration");
        var suggestions = ErrorSuggestions.GetSuggestions(ex);

        Assert.Contains(suggestions, s => s.Contains("yaml") || s.Contains("YAML"));
    }

    [Fact]
    public void GetSuggestions_UnknownError_ReturnsFallback()
    {
        var ex = new Exception("Something completely unexpected happened");
        var suggestions = ErrorSuggestions.GetSuggestions(ex);

        Assert.NotEmpty(suggestions);
        Assert.Contains(suggestions, s => s.Contains("github.com") || s.Contains("Review"));
    }

    [Fact]
    public void GetSuggestions_EvaluateContext_SuggestsModelCheck()
    {
        var ex = new Exception("Something went wrong during evaluation");
        var suggestions = ErrorSuggestions.GetSuggestions(ex, "evaluate");

        Assert.Contains(suggestions, s => s.Contains("mloop list"));
    }

    [Fact]
    public void GetSuggestions_ServeContext_SuggestsPort()
    {
        var ex = new Exception("Address already in use");
        var suggestions = ErrorSuggestions.GetSuggestions(ex, "serve");

        Assert.Contains(suggestions, s => s.Contains("port"));
    }

    [Fact]
    public void GetSuggestions_UpdateContext_SuggestsManualDownload()
    {
        var ex = new Exception("Failed to download latest release");
        var suggestions = ErrorSuggestions.GetSuggestions(ex, "update");

        Assert.Contains(suggestions, s => s.Contains("github.com") || s.Contains("releases") || s.Contains("internet"));
    }

    [Fact]
    public void GetSuggestions_InitContext_DirectoryExists()
    {
        var ex = new Exception("Directory already exists: my-project");
        var suggestions = ErrorSuggestions.GetSuggestions(ex, "init");

        Assert.Contains(suggestions, s => s.Contains("project name") || s.Contains("directory"));
    }

    [Fact]
    public void GetSuggestions_FeatureVectorMismatch_SuggestsSchemaMismatch()
    {
        var ex = new InvalidOperationException(
            "Feature vector dimension mismatch during evaluation. " +
            "The test data's columns don't match the schema the model was trained on " +
            "(a feature column may be missing, renamed, or have a different type).");
        var suggestions = ErrorSuggestions.GetSuggestions(ex, "evaluate");

        // The accurate root cause is a column-structure mismatch, not value-distribution drift:
        // the saved model embeds its fitted featurizers, so dimensions are fixed at training time.
        Assert.Contains(suggestions, s => s.Contains("columns") || s.Contains("schema"));
        Assert.Contains(suggestions, s => s.Contains("type"));
    }

    [Fact]
    public void AddsInformation_OuterAlreadyCarriesInner_IsSuppressed()
    {
        // The wrapping convention "{context}: {inner.Message}" — TrainingEngine's
        // "Training failed for experiment default/exp-001: {inner}" — makes the Inner line a verbatim
        // repeat. With a multi-line diagnosis (class distribution + remediation options) that doubled
        // the whole block on stderr.
        var inner = "Only 1 class has enough samples to train on: 'P' (1 sample)\n  Options: retrain with more rows";
        var outer = $"Training failed for experiment default/exp-001: {inner}";

        Assert.False(ErrorSuggestions.AddsInformation(outer, inner));
    }

    [Fact]
    public void AddsInformation_InnerCarriesNewDetail_IsShown()
    {
        Assert.True(ErrorSuggestions.AddsInformation(
            "Training failed for experiment default/exp-001.",
            "The process cannot access the file because it is being used by another process."));
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    public void AddsInformation_EmptyInner_IsSuppressed(string inner)
    {
        Assert.False(ErrorSuggestions.AddsInformation("Something failed.", inner));
    }

    // --- Resource / no-trial failures (issue: OOM diagnostic contract, cycle-174) ---
    //
    // TrainingEngine rethrows as "Training failed for experiment {id}: {inner.Message}", so every
    // suggestion rule sees the *wrapper*. Anything keyed on the exception type therefore has to walk
    // the inner chain — otherwise the rule silently never fires in production.

    private static Exception AsTrainingEngineWraps(Exception inner) =>
        new InvalidOperationException($"Training failed for experiment default/exp-001: {inner.Message}", inner);

    [Fact]
    public void GetSuggestions_NoSuccessfulTrial_DoesNotAdviseShrinkingTheTimeBudget()
    {
        // The failure IS "the budget bought no finished trial". The generic training fallback used to
        // answer it with "try --time 30" — advice that makes the very failure it is answering worse.
        var ex = AsTrainingEngineWraps(new NoSuccessfulTrialException(900, null));

        var suggestions = ErrorSuggestions.GetSuggestions(ex, "training");

        Assert.DoesNotContain(suggestions, s => s.Contains("--time 30"));
        Assert.DoesNotContain(suggestions, s => s.Contains("smaller time limit"));
    }

    [Fact]
    public void GetSuggestions_RawMlNetTimeout_IsRecognisedByTypeNotWording()
    {
        // The shape actually observed downstream, before AutoMLRunner translates it: ML.NET's own
        // message, which contains neither "timeout" nor "memory", so every message-keyed rule misses it
        // and the generic training fallback answers with "--time 30". Keyed on the type, it can't miss.
        var ex = AsTrainingEngineWraps(new TimeoutException(
            "Training time finished without completing a successful trial. " +
            "Either no trial completed or the metric for all completed trials are NaN or Infinity"));

        var suggestions = ErrorSuggestions.GetSuggestions(ex, "training");

        Assert.DoesNotContain(suggestions, s => s.Contains("--time 30"));
        Assert.DoesNotContain(suggestions, s => s.Contains("smaller time limit"));
        Assert.Contains(suggestions, s => s.Contains("--max-rows") || s.Contains("mloop sample"));
    }

    [Fact]
    public void GetSuggestions_NoSuccessfulTrial_OffersBothCauses()
    {
        var ex = AsTrainingEngineWraps(new NoSuccessfulTrialException(900, null));

        var suggestions = ErrorSuggestions.GetSuggestions(ex, "training");

        // Cause A: budget too small → give more time.
        Assert.Contains(suggestions, s => s.Contains("--time"));
        // Cause B: trials dying on resources → give them less work.
        Assert.Contains(suggestions, s => s.Contains("--max-rows") || s.Contains("mloop sample"));
    }

    [Fact]
    public void GetSuggestions_NoSuccessfulTrial_DoesNotRepeatTheSameRemedyTwice()
    {
        // Caught end-to-end: the no-trial message names memory as a candidate cause, which then matched
        // the memory rule's substring test — printing "--max-rows" and "drop columns" twice each, in
        // slightly different words. Two rules agreeing is not two pieces of advice.
        var ex = AsTrainingEngineWraps(new NoSuccessfulTrialException(900, null));

        var suggestions = ErrorSuggestions.GetSuggestions(ex, "training");

        Assert.Single(suggestions, s => s.Contains("--max-rows"));
        Assert.Single(suggestions, s => s.Contains("olumns"));
    }

    [Fact]
    public void GetSuggestions_WrappedOutOfMemory_SuggestsResourceRemedies()
    {
        // AutoML surfaces OOM through an AggregateException, which AutoMLRunner unwraps; the CLI then
        // wraps it again. The memory rule existed all along but only matched a *bare* OutOfMemoryException.
        var ex = AsTrainingEngineWraps(new OutOfMemoryException("Insufficient memory to continue."));

        var suggestions = ErrorSuggestions.GetSuggestions(ex, "training");

        Assert.Contains(suggestions, s => s.Contains("--max-rows") || s.Contains("mloop sample"));
    }

    [Theory]
    [InlineData("bad allocation")]
    [InlineData("std::bad_alloc")]
    public void GetSuggestions_NativeAllocationFailure_SuggestsResourceRemedies(string nativeText)
    {
        // Native trainers (LightGBM) report exhaustion in their own words; when that text does reach a
        // managed message, it means the same thing as OutOfMemoryException.
        var ex = new InvalidOperationException($"LightGBM training failed: {nativeText}");

        var suggestions = ErrorSuggestions.GetSuggestions(ex, "training");

        Assert.Contains(suggestions, s => s.Contains("--max-rows") || s.Contains("mloop sample"));
    }
}
