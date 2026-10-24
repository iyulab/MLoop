using MLoop.Core.Prediction;
using MLoop.DataStore.Interfaces;

namespace MLoop.CLI.Infrastructure.ML;

/// <summary>
/// Logs predictions that were answered as rows — <c>POST /predict</c> and <c>mloop predict --json --log</c> — and
/// returns the rows stamped with the id each was logged under, the id <c>feedback</c> names. One place, so the two
/// surfaces log the same answer (<see cref="PredictionLogOutput"/>) with the same ids (<see cref="PredictionIds"/>).
/// </summary>
public static class ServedPredictionLog
{
    /// <param name="logger">Where predictions are logged.</param>
    /// <param name="modelName">The model name the rows are logged under.</param>
    /// <param name="experimentId">The experiment that answered.</param>
    /// <param name="inputs">What each row was predicted from, in row order.</param>
    /// <param name="rows">The answers, one per input.</param>
    /// <param name="cancellationToken">Cancels the write.</param>
    public static async Task<IReadOnlyList<PredictionRow>> LogAsync(
        IPredictionLogger logger, string modelName, string experimentId,
        IReadOnlyList<IDictionary<string, object>> inputs, IReadOnlyList<PredictionRow> rows,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(logger);
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(rows);
        if (inputs.Count != rows.Count)
            throw new ArgumentException($"{rows.Count} answers for {inputs.Count} inputs — each answer is logged with its input.", nameof(rows));

        var loggedAt = DateTimeOffset.UtcNow;
        var ids = rows.Select(_ => PredictionIds.New()).ToArray();
        await logger.LogBatchAsync(
            modelName,
            experimentId,
            rows.Select((row, i) => new PredictionLogEntry(
                modelName, experimentId, inputs[i], PredictionLogOutput.Of(row), row.Confidence, loggedAt, ids[i])),
            cancellationToken).ConfigureAwait(false);
        return [.. rows.Select((row, i) => row with { PredictionId = ids[i] })];
    }
}
