using System.Text.Json;
using MLoop.Core.Storage;
using MLoop.Ops.Interfaces;

namespace MLoop.Ops.Services;

/// <summary>
/// Filesystem-based promotion support: backs up the current production model and
/// records promotion history. The promotion copy itself is performed by the
/// authoritative <c>ModelRegistry</c> (CLI/REST); this service reads the current
/// production pointer from the same single authority (<c>production/metadata.json</c>
/// via <see cref="ProductionMetadata"/>) that the registry writes and reads, so all
/// three production-state consumers agree.
/// </summary>
public sealed class FilePromotionManager : IPromotionManager
{
    private readonly string _projectRoot;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        WriteIndented = true
    };

    public FilePromotionManager(string projectRoot)
    {
        _projectRoot = projectRoot ?? throw new ArgumentNullException(nameof(projectRoot));
    }

    public async Task RecordPromotionAsync(
        string modelName,
        string experimentId,
        string? previousExpId,
        string action,
        string? reason = null,
        CancellationToken cancellationToken = default)
    {
        await RecordHistoryAsync(modelName, experimentId, previousExpId, action,
            reason ?? $"{action}: {experimentId} (previous: {previousExpId ?? "none"})",
            cancellationToken).ConfigureAwait(false);
    }

    public async Task<string?> BackupProductionAsync(
        string modelName,
        CancellationToken cancellationToken = default)
    {
        var modelPath = GetModelPath(modelName);
        var productionPath = Path.Combine(modelPath, ExperimentLayout.ProductionDirectory);

        // The presence of the production directory — not a registry file — is what
        // determines whether there is anything to back up. Reading a separate registry
        // for the experiment id used to make this silently skip the backup whenever that
        // file was absent (it always was: ModelRegistry writes registry.json, not the old
        // model-registry.json), risking irrecoverable loss of the previous production model.
        if (!Directory.Exists(productionPath))
            return null;

        var currentExpId = await GetProductionExperimentIdAsync(modelName, cancellationToken).ConfigureAwait(false);

        // Label the backup with the experiment id when known, falling back to a generic
        // prefix so a production directory is still backed up even without metadata.
        var label = string.IsNullOrEmpty(currentExpId) ? "production" : currentExpId;
        var backupPath = Path.Combine(modelPath, "backups",
            $"{label}-{DateTimeOffset.UtcNow:yyyyMMdd-HHmmss}");
        Directory.CreateDirectory(Path.GetDirectoryName(backupPath)!);
        CopyDirectory(productionPath, backupPath);

        return backupPath;
    }

    private async Task RecordHistoryAsync(
        string modelName,
        string experimentId,
        string? previousExpId,
        string action,
        string reason,
        CancellationToken cancellationToken)
    {
        var historyPath = GetHistoryPath(modelName);
        Directory.CreateDirectory(Path.GetDirectoryName(historyPath)!);

        var records = new List<PromotionRecord>();
        if (File.Exists(historyPath))
        {
            var json = await File.ReadAllTextAsync(historyPath, cancellationToken).ConfigureAwait(false);
            records = JsonSerializer.Deserialize<List<PromotionRecord>>(json, JsonOptions)
                ?? new List<PromotionRecord>();
        }

        records.Add(new PromotionRecord(modelName, experimentId, previousExpId, action, reason, DateTimeOffset.UtcNow));

        await File.WriteAllTextAsync(historyPath,
            JsonSerializer.Serialize(records, JsonOptions), cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Reads the current production experiment id from the production model's own
    /// <c>metadata.json</c> (the authoritative artifact ModelRegistry writes on promotion).
    /// Returns null when no production model or metadata exists.
    /// </summary>
    private async Task<string?> GetProductionExperimentIdAsync(
        string modelName,
        CancellationToken cancellationToken)
    {
        var productionDir = Path.Combine(
            GetModelPath(modelName),
            ExperimentLayout.ProductionDirectory);

        var metadata = await ProductionMetadata.ReadAsync(productionDir, cancellationToken).ConfigureAwait(false);
        return metadata?.ExperimentId;
    }

    public static void CopyDirectory(string source, string destination)
    {
        Directory.CreateDirectory(destination);

        foreach (var file in Directory.GetFiles(source))
        {
            File.Copy(file, Path.Combine(destination, Path.GetFileName(file)), overwrite: true);
        }

        foreach (var dir in Directory.GetDirectories(source))
        {
            CopyDirectory(dir, Path.Combine(destination, Path.GetFileName(dir)));
        }
    }

    private string GetModelPath(string modelName)
        => Path.Combine(_projectRoot, "models", modelName.ToLowerInvariant());

    private string GetHistoryPath(string modelName)
        => Path.Combine(GetModelPath(modelName), "promotion-history.json");
}
