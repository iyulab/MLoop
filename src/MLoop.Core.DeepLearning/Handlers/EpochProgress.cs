using System.Diagnostics;
using System.Text.RegularExpressions;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using MLoop.Core.Models;

namespace MLoop.Core.DeepLearning;

/// <summary>
/// Turns a TorchSharp trainer's epoch log lines into training progress, for as long as it is
/// attached to the <see cref="MLContext"/> doing the fit.
/// </summary>
/// <remarks>
/// A deep-learning fit is one call that runs a fixed number of epochs and cannot stop early with a
/// usable model, so neither a trial stream nor the time budget describes it: the progress bar sat at
/// 0% for as long as the fit took — over four minutes on eleven thousand sentence pairs, with nothing
/// to say whether that was a tenth of it or all of it. The trainer does log each epoch it finishes;
/// this reports those against the epoch count the handler chose.
/// </remarks>
public sealed partial class EpochProgress : IDisposable
{
    /// <summary>Epochs a deep-learning trainer runs when the run does not say (<c>training.epochs</c>).</summary>
    public const int DefaultEpochs = 10;

    /// <summary>
    /// Epochs a deep-learning trainer runs for <paramref name="config"/>. Passed to every trainer explicitly, so
    /// the denominator shown is the one actually used rather than a library default restated from memory.
    /// </summary>
    public static int Of(TrainingConfig config) => config.Epochs ?? DefaultEpochs;

    private readonly MLContext _mlContext;
    private readonly IProgress<TrainingProgress>? _progress;
    private readonly string _trainerName;
    private readonly int _epochs;
    private readonly Stopwatch _clock = Stopwatch.StartNew();
    private int _finished;

    private EpochProgress(MLContext mlContext, IProgress<TrainingProgress>? progress, string trainerName, int epochs)
    {
        _mlContext = mlContext;
        _progress = progress;
        _trainerName = trainerName;
        _epochs = epochs;
        _mlContext.Log += OnLog;
    }

    /// <summary>Starts reporting; dispose when the fit returns.</summary>
    public static EpochProgress Attach(
        MLContext mlContext, IProgress<TrainingProgress>? progress, string trainerName, int epochs, Action<string>? log = null)
    {
        log?.Invoke($"{trainerName} trains for {epochs} epochs; the time limit does not shorten it (--epochs sets the count). " +
                    "Progress is reported per epoch.");
        return new EpochProgress(mlContext, progress, trainerName, epochs);
    }

    /// <summary>Epochs finished so far.</summary>
    public int Finished => _finished;

    [GeneratedRegex(@"Finished epoch (\d+)")]
    private static partial Regex FinishedEpoch();

    private void OnLog(object? sender, LoggingEventArgs e)
    {
        if (!FinishedEpoch().IsMatch(e.RawMessage ?? e.Message))
            return;

        // The trainer numbers epochs from zero; what the user wants is how many are done.
        var done = Interlocked.Increment(ref _finished);
        _progress?.Report(new TrainingProgress
        {
            Phase = TrainingPhase.Epoch,
            Epoch = Math.Min(done, _epochs),
            MaxEpochs = _epochs,
            TrialNumber = 0,
            TrainerName = _trainerName,
            Metric = 0,
            MetricName = "",
            ElapsedSeconds = _clock.Elapsed.TotalSeconds
        });
    }

    public void Dispose() => _mlContext.Log -= OnLog;
}
