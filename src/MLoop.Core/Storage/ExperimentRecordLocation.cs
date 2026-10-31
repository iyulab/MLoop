using System.Text.Json;

namespace MLoop.Core.Storage;

/// <summary>
/// Where the experiment record of a model file is — the folder holding the columns the model was trained on, its label
/// and task. A model file carries none of that; <c>config.json</c> and <c>metadata.json</c> beside it do. Predicting with
/// a file named by its path used to look the record up by the name of the file's folder in the <em>current</em> project,
/// so a model copied out of its experiment folder, and the file in the <c>production</c> slot (which holds a pointer
/// to the experiment, not the record), were scored as the data reads: labels came back as <c>True</c>/<c>False</c> where
/// the model was trained on <c>yes</c>/<c>no</c>, and a categorical column failed. A folder of another project that
/// happened to share an experiment id took the current project's record.
/// </summary>
public static class ExperimentRecordLocation
{
    /// <summary>An experiment record: its folder and the experiment's id.</summary>
    /// <param name="Directory">The folder holding <c>config.json</c> and <c>metadata.json</c>.</param>
    /// <param name="ExperimentId">The id the record gives itself (<c>metadata.json</c>), else its folder's name.</param>
    public sealed record Found(string Directory, string ExperimentId);

    /// <summary>
    /// The record of the model at <paramref name="modelFile"/>: the folder it sits in when that is an experiment folder, or —
    /// for the file in a model's <c>production</c> slot — the experiment the slot points at, in the <c>staging</c> folder
    /// beside it. <see langword="null"/> when neither exists (a bare model file, a pointer to an experiment that is gone).
    /// </summary>
    public static async Task<Found?> FindAsync(string modelFile, CancellationToken cancellationToken = default)
    {
        var directory = Path.GetDirectoryName(Path.GetFullPath(modelFile));
        if (directory is null)
            return null;

        if (IsRecord(directory))
            return new Found(directory, IdOf(directory) ?? Path.GetFileName(directory));

        if (!string.Equals(Path.GetFileName(directory), ExperimentLayout.ProductionDirectory, StringComparison.Ordinal))
            return null;

        var slot = await ProductionMetadata.ReadAsync(directory, cancellationToken).ConfigureAwait(false);
        // The id comes from a file the user may have been handed: it names a folder, never a path.
        if (slot?.ExperimentId is not { Length: > 0 } id || id != Path.GetFileName(id) || id is "." or "..")
            return null;
        var staged = Directory.GetParent(directory) is { } model
            ? Path.Combine(model.FullName, ExperimentLayout.StagingDirectory, id)
            : null;
        return staged is not null && IsRecord(staged) ? new Found(staged, id) : null;
    }

    private static bool IsRecord(string directory) =>
        File.Exists(Path.Combine(directory, ExperimentLayout.ConfigFileName))
        && File.Exists(Path.Combine(directory, ExperimentLayout.MetadataFileName));

    /// <summary>The <c>experimentId</c> a record's <c>metadata.json</c> gives; null when it does not say.</summary>
    private static string? IdOf(string directory)
    {
        try
        {
            using var stream = File.OpenRead(Path.Combine(directory, ExperimentLayout.MetadataFileName));
            using var document = JsonDocument.Parse(stream);
            foreach (var property in document.RootElement.EnumerateObject())
            {
                if (string.Equals(property.Name, "experimentId", StringComparison.OrdinalIgnoreCase)
                    && property.Value.ValueKind == JsonValueKind.String
                    && property.Value.GetString() is { Length: > 0 } id
                    && id == Path.GetFileName(id))
                    return id;
            }
        }
        catch (Exception e) when (e is IOException or JsonException or InvalidOperationException)
        {
            // an unreadable metadata.json is no id; the folder's name stands in
        }
        return null;
    }
}
