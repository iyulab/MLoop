using MLoop.Core.DeepLearning;

namespace MLoop.Core.DeepLearning.Tests.Handlers;

public class PretrainedWeightsTests
{
    // The shape ML.NET's TorchSharp trainers throw when their weights cannot be fetched
    // (ResourceManagerUtils.GetErrorMessage + the trainer's own suffix), as observed.
    private const string Url = "https://aka.ms/mlnet-resources/models/pretrained_Roberta_encoder.tsm";
    private static readonly Exception DownloadFailure = new InvalidOperationException(
        $"Error downloading resource from '{Url}': DownloadFailed with exception One or more errors occurred. "
        + "(The wait completed due to an abandoned mutex.)\n\nmodel file could not be downloaded!");

    [Fact]
    public void A_failed_download_is_told_with_the_file_the_folder_and_how_to_fetch_it()
    {
        var ex = Assert.Throws<PretrainedWeightsUnavailableException>(() =>
            PretrainedWeights.Guard<int>(() => throw new AggregateException(DownloadFailure)));

        var cache = Path.Combine(Path.GetTempPath(), "mlnet");
        Assert.Contains(Url, ex.Message);
        Assert.Contains(Path.Combine(cache, "pretrained_Roberta_encoder.tsm"), ex.Message);
        Assert.Contains("curl -L -o", ex.Message);
        Assert.Contains("dotnet/machinelearning#6980", ex.Message);
        Assert.Same(DownloadFailure, ex.InnerException?.InnerException);
    }

    [Fact]
    public void Any_other_failure_passes_through_untouched()
    {
        var other = new InvalidOperationException("Could not find input column 'context'");

        var ex = Assert.Throws<InvalidOperationException>(() => PretrainedWeights.Guard<int>(() => throw other));

        Assert.Same(other, ex);
    }

    [Fact]
    public void A_result_passes_through() => Assert.Equal(7, PretrainedWeights.Guard(() => 7));
}
