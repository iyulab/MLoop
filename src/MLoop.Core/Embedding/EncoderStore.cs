using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace MLoop.Core.Embedding;

/// <summary>An encoder on disk: its definition and the graph file it runs.</summary>
public sealed record InstalledEncoder(EncoderDefinition Definition, string ModelPath);

/// <summary>
/// The encoders installed on this machine — one directory per encoder under <c>~/.mloop/encoders/</c> holding its
/// definition (<c>encoder.json</c>) and its graph (<c>model.onnx</c>). Outside any project: an encoder is not trained
/// and serves every project the same way, like an on-demand runtime.
/// </summary>
public sealed class EncoderStore
{
    /// <summary>The definition file in an encoder's directory.</summary>
    public const string DefinitionFileName = "encoder.json";

    /// <summary>The graph file in an encoder's directory.</summary>
    public const string ModelFileName = "model.onnx";

    private static readonly JsonSerializerOptions Json = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true,
        Converters = { new JsonStringEnumConverter(JsonNamingPolicy.KebabCaseLower) },
    };

    /// <summary>The environment variable that moves <see cref="DefaultRoot"/> — a container's volume, a test's directory.</summary>
    public const string RootVariable = "MLOOP_ENCODERS";

    /// <summary>Where encoders are kept unless told otherwise: <see cref="RootVariable"/>, else <c>~/.mloop/encoders</c>.</summary>
    public static string DefaultRoot =>
        Environment.GetEnvironmentVariable(RootVariable) is { Length: > 0 } root
            ? root
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".mloop", "encoders");

    /// <param name="root">The directory encoders live under; <see langword="null"/> for <see cref="DefaultRoot"/>.</param>
    public EncoderStore(string? root = null) => Root = root ?? DefaultRoot;

    /// <summary>The directory encoders live under.</summary>
    public string Root { get; }

    /// <summary>Every installed encoder, by id. A directory without both files is not an encoder and is skipped.</summary>
    public IReadOnlyList<InstalledEncoder> List()
    {
        if (!Directory.Exists(Root))
            return [];
        return [.. Directory.EnumerateDirectories(Root)
            .Select(dir => Read(Path.GetFileName(dir)))
            .OfType<InstalledEncoder>()
            .OrderBy(e => e.Definition.Id, StringComparer.Ordinal)];
    }

    /// <summary>The installed encoder <paramref name="id"/>, or <see langword="null"/>.</summary>
    public InstalledEncoder? Find(string id) => IsValidId(id) ? Read(id) : null;

    /// <summary>
    /// The encoder a request means: <paramref name="id"/> when given, else the only one installed. <see langword="null"/>
    /// with the reason when there is none to use.
    /// </summary>
    public (InstalledEncoder? Encoder, string? Refusal) Resolve(string? id)
    {
        if (!string.IsNullOrWhiteSpace(id))
        {
            return Find(id.Trim()) is { } found
                ? (found, null)
                : (null, $"Encoder '{id.Trim()}' is not installed. Installed: {Names()}. {HowToInstall(id)}");
        }
        var all = List();
        return all.Count switch
        {
            1 => (all[0], null),
            0 => (null, $"No encoder is installed. {HowToInstall(null)}"),
            _ => (null, $"Several encoders are installed ({Names()}); name one."),
        };
    }

    /// <summary>
    /// Installs the graph at <paramref name="modelPath"/> as <paramref name="definition"/>, recording the file's SHA-256 —
    /// which, with the id, is the encoder's identity for whoever keeps its vectors. A definition that pins a SHA-256 the
    /// file does not have is refused. Replaces an encoder of the same id.
    /// </summary>
    public InstalledEncoder Add(EncoderDefinition definition, string modelPath)
    {
        ArgumentNullException.ThrowIfNull(definition);
        if (!IsValidId(definition.Id))
            throw new ArgumentException($"'{definition.Id}' is not an encoder id: use letters, digits, '-', '_' and '.'.", nameof(definition));
        if (definition.Problem() is { } problem)
            throw new ArgumentException($"Encoder '{definition.Id}' cannot run: {problem}.", nameof(definition));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Encoder graph not found: {modelPath}", modelPath);

        var sha = Sha256Of(modelPath);
        if (definition.Sha256 is { } pinned && !pinned.Equals(sha, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException($"'{modelPath}' has SHA-256 {sha}; encoder '{definition.Id}' pins {pinned}.");

        var dir = Path.Combine(Root, definition.Id);
        var staging = dir + ".installing";
        if (Directory.Exists(staging))
            Directory.Delete(staging, recursive: true);
        Directory.CreateDirectory(staging);
        File.Copy(modelPath, Path.Combine(staging, ModelFileName));
        var recorded = definition with { Sha256 = sha };
        File.WriteAllText(Path.Combine(staging, DefinitionFileName), JsonSerializer.Serialize(recorded, Json));
        if (Directory.Exists(dir))
            Directory.Delete(dir, recursive: true);
        Directory.Move(staging, dir);
        return new InstalledEncoder(recorded, Path.Combine(dir, ModelFileName));
    }

    /// <summary>
    /// Downloads a catalog entry's weights (<see cref="EncoderDefinition.Url"/>) and installs them, refusing a file
    /// whose SHA-256 is not the one the entry pins. The download lands beside the encoders, so a partial one never
    /// takes the place of an installed encoder, and is removed whatever happens.
    /// </summary>
    /// <param name="definition">The entry — with a URL and a SHA-256 (<see cref="EncoderCatalog"/>).</param>
    /// <param name="http">The client to download with.</param>
    /// <param name="progress">Bytes downloaded so far, and the total when the server says it.</param>
    /// <param name="cancellationToken">Stops the download.</param>
    public async Task<InstalledEncoder> InstallAsync(
        EncoderDefinition definition, HttpClient http,
        IProgress<(long Downloaded, long? Total)>? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(definition);
        ArgumentNullException.ThrowIfNull(http);
        if (definition.Url is not { Length: > 0 } url || definition.Sha256 is not { Length: > 0 } pinned)
            throw new ArgumentException($"Encoder '{definition.Id}' has no pinned download (URL and SHA-256); add it from a file instead.", nameof(definition));
        if (!IsValidId(definition.Id))
            throw new ArgumentException($"'{definition.Id}' is not an encoder id.", nameof(definition));

        Directory.CreateDirectory(Root);
        var download = Path.Combine(Root, definition.Id + ".download");
        try
        {
            using (var response = await http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false))
            {
                response.EnsureSuccessStatusCode();
                var total = response.Content.Headers.ContentLength;
                await using var source = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
                await using var target = new FileStream(download, FileMode.Create, FileAccess.Write, FileShare.None, 81920, useAsync: true);
                var buffer = new byte[81920];
                long downloaded = 0;
                int read;
                while ((read = await source.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
                {
                    await target.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
                    downloaded += read;
                    progress?.Report((downloaded, total));
                }
            }

            var sha = Sha256Of(download);
            if (!sha.Equals(pinned, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException($"The download of encoder '{definition.Id}' has SHA-256 {sha}, not the pinned {pinned}; nothing was installed.");
            return Add(definition, download);
        }
        finally
        {
            if (File.Exists(download))
                File.Delete(download);
        }
    }

    /// <summary>Removes encoder <paramref name="id"/>; false when it was not installed.</summary>
    public bool Remove(string id)
    {
        if (!IsValidId(id))
            return false;
        var dir = Path.Combine(Root, id);
        if (!Directory.Exists(dir))
            return false;
        Directory.Delete(dir, recursive: true);
        return true;
    }

    /// <summary>Reads an encoder definition file — what <c>mloop encoder add</c> takes.</summary>
    public static EncoderDefinition ReadDefinition(string path) =>
        JsonSerializer.Deserialize<EncoderDefinition>(File.ReadAllText(path), Json)
        ?? throw new InvalidDataException($"'{path}' holds no encoder definition.");

    private InstalledEncoder? Read(string id)
    {
        var dir = Path.Combine(Root, id);
        var definitionPath = Path.Combine(dir, DefinitionFileName);
        var modelPath = Path.Combine(dir, ModelFileName);
        if (!File.Exists(definitionPath) || !File.Exists(modelPath))
            return null;
        try
        {
            var definition = WithCatalogPatches(ReadDefinition(definitionPath));
            return definition.Id == id ? new InstalledEncoder(definition, modelPath) : null;
        }
        catch (JsonException)
        {
            return null;
        }
    }

    // An install recorded before the catalog declared patch sizes: the same weights are the same graph, so the catalog's
    // patch size holds for it. Any other definition keeps what it says.
    private static EncoderDefinition WithCatalogPatches(EncoderDefinition definition) =>
        definition.PatchSize is null && definition.Sha256 is not null
        && EncoderCatalog.All.FirstOrDefault(e => e.Sha256 == definition.Sha256) is { PatchSize: { } patch }
            ? definition with { PatchSize = patch }
            : definition;

    // The catalog's command when the catalog names this encoder (or, with no name, any); adding a file otherwise.
    private static string HowToInstall(string? id)
    {
        if (EncoderCatalog.Find(id) is { } entry)
            return $"Install it with: mloop encoder install {entry.Id}";
        return id is null
            ? $"Install one with: mloop encoder install <{EncoderCatalog.Ids}>, or add your own: mloop encoder add <encoder.json> --model <model.onnx>"
            : "Add it with: mloop encoder add <encoder.json> --model <model.onnx>";
    }

    private string Names()
    {
        var all = List();
        return all.Count == 0 ? "none" : string.Join(", ", all.Select(e => e.Definition.Id));
    }

    private static bool IsValidId(string? id) =>
        !string.IsNullOrWhiteSpace(id) && id is not ("." or "..")
        && id.All(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '_' or '.');

    private static string Sha256Of(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }
}
