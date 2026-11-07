using MLoop.Core.Storage;

namespace MLoop.Core.Models;

/// <summary>
/// Whether the model in a production slot is the file the promotion put there. A promotion records the SHA-256 of what it copied
/// (<see cref="ProductionMetadata.ModelSha256"/>); a later load compares the file with it, so a model that changed afterwards - a damaged
/// disk, a file written over by hand, another tool - is said so, instead of answering as if it were the one that passed the gate.
/// </summary>
/// <remarks>
/// A statement, not a refusal: whether to refuse a changed model is a decision about hand-placed models that has not been made. The hash is
/// the one every answer reports as <c>modelSha256</c>, remembered per file while its size and write time stay the same
/// (<see cref="ModelFingerprint"/>), so a prediction pays it once. Never throws: a slot without a record, an experiment folder, an
/// unreadable pointer all read as "nothing to compare" - integrity is judged on what was recorded, and a check must not break serving.
/// </remarks>
public static class ModelIntegrity
{
    /// <summary>
    /// A sentence saying the model at <paramref name="modelPath"/> is not the one its promotion recorded, or <c>null</c> when it is, when it
    /// is not a production slot's <c>model.zip</c>, or when the slot was promoted before the hash was recorded.
    /// </summary>
    public static string? Mismatch(string modelPath)
    {
        try
        {
            var full = Path.GetFullPath(modelPath);
            var directory = Path.GetDirectoryName(full);
            if (directory is null
                || !string.Equals(Path.GetFileName(directory), ExperimentLayout.ProductionDirectory, StringComparison.Ordinal)
                || !string.Equals(Path.GetFileName(full), ExperimentLayout.ModelFileName, StringComparison.Ordinal)
                || !File.Exists(full))
                return null;

            var recorded = ProductionMetadata.ReadAsync(directory).GetAwaiter().GetResult()?.ModelSha256;
            if (string.IsNullOrWhiteSpace(recorded))
                return null;

            var actual = ModelFingerprint.Sha256(full);
            return string.Equals(recorded, actual, StringComparison.OrdinalIgnoreCase)
                ? null
                : $"The production model '{full}' is not the file its promotion recorded (sha256 {Short(recorded)} then, {Short(actual)} now): it was changed after it was promoted. "
                  + "'mloop promote <experiment-id>' puts a staged model there again and 'mloop rollback' returns to the previous one.";
        }
        catch (Exception)
        {
            return null;
        }
    }

    private static string Short(string sha256) => sha256.Length > 12 ? sha256[..12] : sha256;
}
