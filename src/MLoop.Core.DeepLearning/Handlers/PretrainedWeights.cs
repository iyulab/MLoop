using System.Text.RegularExpressions;

namespace MLoop.Core.DeepLearning;

/// <summary>
/// What a user is told when ML.NET cannot fetch the pretrained weights a deep-learning task trains
/// from.
/// </summary>
/// <remarks>
/// <para>
/// The TorchSharp trainers (text classification, sentence similarity, NER, question answering,
/// object detection) download their weights on first use into <c>mlnet</c> under the context's temp
/// folder — <see cref="Path.GetTempPath"/>, since MLoop does not set one — and reuse the file after
/// that. The download fails on this project's machines with "The wait completed due to an abandoned
/// mutex": ML.NET's downloader takes a named mutex on one thread and releases it on another, an open
/// upstream defect (dotnet/machinelearning#6980). Large files hit it reliably (measured twice on the
/// 496MB RoBERTa encoder), and a changed TEMP hits it again because the cache follows TEMP.
/// </para>
/// <para>
/// ML.NET's own message names the URL but not where the file is expected, and MLoop's generic advice
/// after it suggested commands unrelated to the failure. Fetching the file into that folder by hand is
/// enough — the trainer finds it and skips the download.
/// </para>
/// </remarks>
public static partial class PretrainedWeights
{
    private const string DownloadFailureMarker = "Error downloading resource";

    /// <summary>The folder the trainers look for their weights in.</summary>
    public static string CacheDirectory => Path.Combine(Path.GetTempPath(), "mlnet");

    /// <summary>
    /// Runs <paramref name="fit"/>, rethrowing a failed weights download as a
    /// <see cref="PretrainedWeightsUnavailableException"/> that says what to fetch and where to put it.
    /// Any other failure passes through untouched.
    /// </summary>
    public static T Guard<T>(Func<T> fit)
    {
        try
        {
            return fit();
        }
        catch (Exception ex) when (FindDownloadFailure(ex) is { } failure)
        {
            throw new PretrainedWeightsUnavailableException(Explain(failure.Message), ex);
        }
    }

    private static Exception? FindDownloadFailure(Exception? ex)
    {
        for (; ex != null; ex = ex.InnerException)
        {
            if (ex.Message.Contains(DownloadFailureMarker, StringComparison.Ordinal))
                return ex;
            if (ex is AggregateException aggregate)
                foreach (var inner in aggregate.InnerExceptions)
                    if (FindDownloadFailure(inner) is { } found)
                        return found;
        }
        return null;
    }

    private static string Explain(string mlnetMessage)
    {
        const string cause =
            "ML.NET could not download the pretrained weights this task trains from. Its downloader fails on large "
            + "files with an abandoned-mutex error (dotnet/machinelearning#6980).";

        if (UrlPattern().Match(mlnetMessage) is not { Success: true } match)
            return $"{cause} Fetch the file named in the error below into '{CacheDirectory}' and train again: {mlnetMessage}";

        var url = match.Groups[1].Value;
        var file = Path.Combine(CacheDirectory, url.Split('/').Last());
        return $"{cause} Download it yourself and training will use it:\n"
            + $"  curl -L -o \"{file}\" {url}\n"
            + $"The file is looked for in '{CacheDirectory}', the temp folder — a different TEMP looks somewhere else.";
    }

    [GeneratedRegex("from '([^']+)'")]
    private static partial Regex UrlPattern();
}

/// <summary>
/// The pretrained weights a deep-learning task trains from could not be fetched. The message says
/// which file, where it is looked for and how to fetch it — a typed failure so the CLI answers with
/// that remedy rather than its generic training advice.
/// </summary>
public sealed class PretrainedWeightsUnavailableException(string message, Exception inner)
    : InvalidOperationException(message, inner);
