using System.Globalization;
using System.Text.RegularExpressions;

namespace MLoop.Core.DeepLearning;

/// <summary>
/// The loss a TorchSharp trainer logs at every step it is asked to (<c>Row: 12, Loss: 1.5576</c>), gathered into a mean per epoch.
/// </summary>
/// <remarks>
/// A fit that finds nothing could not be told from one that finds boxes it is not confident about: the training log said how many
/// epochs had finished and nothing of whether the loss was falling. The trainer logs the loss every <c>LogEveryNStep</c> steps (50 by
/// default, so a small set logged none); it is set to every step and each epoch's mean is written beside the epoch count, which
/// shows a loss that falls, one that sits, and one that blows up. Epoch 1 is mostly the first steps' loss falling from a large
/// start, so what reads is the epochs after it.
/// </remarks>
public sealed partial class EpochLoss
{
    [GeneratedRegex(@"Row: \d+, Loss: (?<loss>\S+)")]
    private static partial Regex StepLine();

    private readonly object _gate = new();
    private readonly List<double> _epochMeans = [];
    private double _sum;
    private int _steps;

    /// <summary>The mean loss of each epoch finished so far, in order.</summary>
    public IReadOnlyList<double> EpochMeans
    {
        get { lock (_gate) return [.. _epochMeans]; }
    }

    /// <summary>Reads the loss out of a trainer's step line; false for any other message, or a loss that is not a finite number.</summary>
    public static bool TryParseStep(string? message, out double loss)
    {
        loss = 0;
        if (message is null || StepLine().Match(message) is not { Success: true } match)
            return false;

        return double.TryParse(match.Groups["loss"].Value, NumberStyles.Float, CultureInfo.InvariantCulture, out loss) && double.IsFinite(loss);
    }

    /// <summary>Counts one step's loss toward the epoch in progress.</summary>
    public void AddStep(double loss)
    {
        lock (_gate)
        {
            _sum += loss;
            _steps++;
        }
    }

    /// <summary>
    /// Closes the epoch in progress: its mean loss and the steps it was the mean of, or <see langword="null"/> when the trainer
    /// logged no step in it.
    /// </summary>
    public (double Mean, int Steps)? FinishEpoch()
    {
        lock (_gate)
        {
            if (_steps == 0)
                return null;

            var result = (Mean: _sum / _steps, Steps: _steps);
            _epochMeans.Add(result.Mean);
            _sum = 0;
            _steps = 0;
            return result;
        }
    }

    /// <summary>How an epoch's line reads in the training log.</summary>
    public static string Describe(string trainerName, int epoch, int epochs, double mean, int steps) =>
        string.Create(CultureInfo.InvariantCulture, $"{trainerName} epoch {epoch}/{epochs}: loss {mean:G4} (mean of {steps} steps)");
}
