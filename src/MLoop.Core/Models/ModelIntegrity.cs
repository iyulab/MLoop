using MLoop.Core.Storage;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace MLoop.Core.Models;

/// <summary>
/// Whether the model in a production slot is the file the promotion put there. A promotion records the SHA-256 of what it copied
/// (<see cref="ProductionMetadata.ModelSha256"/>); a later load compares the file with it, so a model that changed afterwards - a damaged
/// disk, a file written over by hand, another tool - is said so, instead of answering as if it were the one that passed the gate.
/// </summary>
/// <remarks>
/// A statement by default, a refusal when the project asks for one: <c>integrity: refuse_changed_model: true</c> in its mloop.yaml
/// (<see cref="IntegritySettings"/>, <see cref="Enforce"/>) — placing a model by hand is a way some projects work, so the answer to a changed
/// model is the project's to give, and its absence is the old behaviour. The hash is
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

    /// <summary>
    /// The sentence to say when the model changed and the project did not ask for a refusal: <see cref="Mismatch"/>, or <c>null</c> when
    /// there is nothing to say or <see cref="Enforce"/> is about to refuse (which says it instead, once).
    /// </summary>
    public static string? Warning(string modelPath)
        => Mismatch(modelPath) is { } changed && !RefusesChangedModel(modelPath) ? changed : null;

    /// <summary>
    /// Refuses a production model that changed after its promotion when the project's mloop.yaml sets
    /// <c>integrity: refuse_changed_model: true</c>. Every scoring load passes here (<see cref="ScoringModel.Load(Microsoft.ML.MLContext, string, string?, ScoringLayers)"/>),
    /// so no path scores a model the project said not to trust. Nothing to refuse — a matching hash, a slot without a record, a model outside
    /// the production slot — returns silently.
    /// </summary>
    /// <exception cref="ChangedModelException">The model changed after it was promoted and the project refuses that.</exception>
    public static void Enforce(string modelPath)
    {
        if (Mismatch(modelPath) is { } changed && RefusesChangedModel(modelPath))
            throw new ChangedModelException(changed + " It is refused because 'integrity: refuse_changed_model' is on in mloop.yaml.");
    }

    /// <summary>
    /// Whether the project the model belongs to refuses a changed production model. The project file is found above the slot
    /// (<c>models/&lt;name&gt;/production</c> sits two levels under it) and read only here, when a mismatch exists — a matching model never
    /// pays for it. An unreadable or absent file is "not refused": a check that must not break serving does not turn a typo into an outage.
    /// </summary>
    public static bool RefusesChangedModel(string modelPath)
    {
        try
        {
            var slot = Path.GetDirectoryName(Path.GetFullPath(modelPath));
            for (var directory = slot is null ? null : Path.GetDirectoryName(slot); directory is not null; directory = Path.GetDirectoryName(directory))
            {
                var file = Path.Combine(directory, ProjectFileName);
                if (!File.Exists(file))
                    continue;
                return ProjectFileReader.Deserialize<ProjectIntegrity>(File.ReadAllText(file)).Integrity?.RefuseChangedModel == true;
            }
            return false;
        }
        catch (Exception)
        {
            return false;
        }
    }

    /// <summary>The project file's name — the same file every command reads its configuration from.</summary>
    public const string ProjectFileName = "mloop.yaml";

    private static readonly IDeserializer ProjectFileReader = new DeserializerBuilder()
        .WithNamingConvention(UnderscoredNamingConvention.Instance)
        .IgnoreUnmatchedProperties()
        .Build();

    private sealed class ProjectIntegrity
    {
        public IntegritySettings? Integrity { get; set; }
    }

    private static string Short(string sha256) => sha256.Length > 12 ? sha256[..12] : sha256;
}
