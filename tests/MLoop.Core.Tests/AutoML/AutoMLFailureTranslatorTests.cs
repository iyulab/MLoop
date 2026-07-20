using MLoop.Core.AutoML;

namespace MLoop.Core.Tests.AutoML;

/// <summary>
/// Pins the terminal-exception shapes measured against Microsoft.ML.AutoML 0.23.0 (cycle-173):
/// AutoML wraps its own failures in an <see cref="AggregateException"/>, and the "no trial completed"
/// case arrives as an inner <see cref="TimeoutException"/> — identifiable by type, not by its
/// (ML.NET-owned, localizable) message text.
/// </summary>
public class AutoMLFailureTranslatorTests
{
    private const string MlNetNoTrialMessage =
        "Training time finished without completing a successful trial. " +
        "Either no trial completed or the metric for all completed trials are NaN or Infinity";

    [Fact]
    public void TryTranslate_AggregateWrappingTimeout_YieldsNoSuccessfulTrial()
    {
        var actual = new AggregateException(new TimeoutException(MlNetNoTrialMessage));

        Assert.True(AutoMLFailureTranslator.TryTranslate(actual, 900, out var translated));

        var noTrial = Assert.IsType<NoSuccessfulTrialException>(translated);
        Assert.Equal(900, noTrial.TimeLimitSeconds);
        Assert.Same(actual, noTrial.InnerException);
    }

    [Fact]
    public void TryTranslate_NoSuccessfulTrialMessage_NamesBothCausesNotOne()
    {
        AutoMLFailureTranslator.TryTranslate(
            new AggregateException(new TimeoutException(MlNetNoTrialMessage)), 900, out var translated);

        // MLoop cannot distinguish "budget too small" from "every trial died" here — AutoML's
        // progressHandler reports nothing when no trial completes. Claiming one would be a guess.
        Assert.Contains("budget", translated.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("every trial failed", translated.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void TryTranslate_BareTimeout_YieldsNoSuccessfulTrial()
    {
        // Defensive: the same condition without AutoML's threading wrapper.
        Assert.True(AutoMLFailureTranslator.TryTranslate(
            new TimeoutException(MlNetNoTrialMessage), 60, out var translated));

        Assert.IsType<NoSuccessfulTrialException>(translated);
    }

    [Fact]
    public void TryTranslate_AggregateWrappingOutOfMemory_UnwrapsToOutOfMemory()
    {
        // Unwrapping is the whole fix here: the downstream suggestion layer already knows what to say
        // about OutOfMemoryException — it just never saw one through the AggregateException wrapper.
        var oom = new OutOfMemoryException("Exception of type 'System.OutOfMemoryException' was thrown.");

        Assert.True(AutoMLFailureTranslator.TryTranslate(new AggregateException(oom), 900, out var translated));

        Assert.Same(oom, translated);
    }

    [Fact]
    public void TryTranslate_AucUndefinedFailure_IsLeftAlone()
    {
        // BUG-22/24/36: AutoMLRunner recovers from this by falling back to another metric. Translating
        // it would break that recovery, so the translator must not claim it.
        var auc = new AggregateException(
            new ArgumentOutOfRangeException("PosSample", "AUC is not defined when there is no positive class in the data"));

        Assert.False(AutoMLFailureTranslator.TryTranslate(auc, 900, out var translated));
        Assert.Same(auc, translated);
    }

    [Fact]
    public void TryTranslate_UnrelatedFailure_IsLeftAlone()
    {
        var other = new InvalidOperationException("Schema mismatch for feature column 'Features'");

        Assert.False(AutoMLFailureTranslator.TryTranslate(other, 900, out var translated));
        Assert.Same(other, translated);
    }
}
