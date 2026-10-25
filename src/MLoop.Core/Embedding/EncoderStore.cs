using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace MLoop.Core.Embedding;

/// <summary>An encoder on disk: its definition, the graph file it runs and — a text encoder's — its tokenizer model.</summary>
public sealed record InstalledEncoder(EncoderDefinition Definition, string ModelPath, string? TokenizerPath = null);

/// <summary>
/// The encoders installed on this machine — one directory per encoder under <c>~/.mloop/encoders/</c> holding its
/// definition (<c>encoder.json</c>), its graph (<c>model.onnx</c>) and, for text, its tokenizer model
/// (<c>tokenizer.model</c>). Outside any project: an encoder is not trained
/// and serves every project the same way, like an on-demand runtime.
/// </summary>
public sealed class EncoderStore
{
    /// <summary>The definition file in an encoder's directory.</summary>
    public const string DefinitionFileName = "encoder.json";

    /// <summary>The graph file in an encoder's directory.</summary>
    public const string ModelFileName = "model.onnx";

    /// <summary>A text encoder's tokenizer model (SentencePiece) in its directory.</summary>
    public const string TokenizerFileName = "tokenizer.model";

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
    /// The encoder of <paramref name="kind"/> a request means: <paramref name="id"/> when given, else the only one of that
    /// kind installed. <see langword="null"/> with the reason when there is none to use — an encoder named that reads the
    /// other kind included.
    /// </summary>
    public (InstalledEncoder? Encoder, string? Refusal) Resolve(string? id, EncoderKind kind = EncoderKind.Image)
    {
        var noun = Noun(kind);
        if (!string.IsNullOrWhiteSpace(id))
        {
            if (Find(id.Trim()) is not { } found)
                return (null, $"Encoder '{id.Trim()}' is not installed. Installed: {Names()}. {HowToInstall(id, kind)}");
            return found.Definition.Kind == kind
                ? (found, null)
                : (null, $"Encoder '{found.Definition.Id}' reads {Noun(found.Definition.Kind)}, not {noun}.");
        }
        var all = List().Where(e => e.Definition.Kind == kind).ToList();
        return all.Count switch
        {
            1 => (all[0], null),
            0 => (null, $"No encoder of {noun} is installed. {HowToInstall(null, kind)}"),
            _ => (null, $"Several encoders of {noun} are installed ({string.Join(", ", all.Select(e => e.Definition.Id))}); name one."),
        };
    }

    /// <summary>
    /// Installs the graph at <paramref name="modelPath"/> — and a text encoder's tokenizer model at
    /// <paramref name="tokenizerPath"/> — as <paramref name="definition"/>, recording each file's SHA-256: with the id, the
    /// encoder's identity for whoever keeps its vectors. A definition that pins a SHA-256 a file does not have is refused.
    /// Replaces an encoder of the same id.
    /// </summary>
    public InstalledEncoder Add(EncoderDefinition definition, string modelPath, string? tokenizerPath = null)
    {
        ArgumentNullException.ThrowIfNull(definition);
        if (!IsValidId(definition.Id))
            throw new ArgumentException($"'{definition.Id}' is not an encoder id: use letters, digits, '-', '_' and '.'.", nameof(definition));
        if (definition.Problem() is { } problem)
            throw new ArgumentException($"Encoder '{definition.Id}' cannot run: {problem}.", nameof(definition));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Encoder graph not found: {modelPath}", modelPath);
        var text = definition.Kind == EncoderKind.Text;
        if (text && tokenizerPath is null)
            throw new ArgumentException($"Encoder '{definition.Id}' reads text: it needs its tokenizer model (SentencePiece) beside the graph.", nameof(tokenizerPath));
        if (!text && tokenizerPath is not null)
            throw new ArgumentException($"Encoder '{definition.Id}' reads images: it takes no tokenizer model.", nameof(tokenizerPath));
        if (tokenizerPath is not null && !File.Exists(tokenizerPath))
            throw new FileNotFoundException($"Tokenizer model not found: {tokenizerPath}", tokenizerPath);

        var sha = Pinned(modelPath, definition.Sha256, definition.Id);
        var tokenizerSha = tokenizerPath is null ? null : Pinned(tokenizerPath, definition.TokenizerSha256, definition.Id);

        var dir = Path.Combine(Root, definition.Id);
        var staging = dir + ".installing";
        if (Directory.Exists(staging))
            Directory.Delete(staging, recursive: true);
        Directory.CreateDirectory(staging);
        File.Copy(modelPath, Path.Combine(staging, ModelFileName));
        if (tokenizerPath is not null)
            File.Copy(tokenizerPath, Path.Combine(staging, TokenizerFileName));
        var recorded = definition with { Sha256 = sha, TokenizerSha256 = tokenizerSha };
        File.WriteAllText(Path.Combine(staging, DefinitionFileName), JsonSerializer.Serialize(recorded, Json));
        if (Directory.Exists(dir))
            Directory.Delete(dir, recursive: true);
        Directory.Move(staging, dir);
        return new InstalledEncoder(recorded, Path.Combine(dir, ModelFileName), tokenizerPath is null ? null : Path.Combine(dir, TokenizerFileName));
    }

    // The file's SHA-256 — refused when the definition pins another.
    private static string Pinned(string path, string? pinned, string id)
    {
        var sha = Sha256Of(path);
        if (pinned is not null && !pinned.Equals(sha, StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException($"'{path}' has SHA-256 {sha}; encoder '{id}' pins {pinned}.");
        return sha;
    }

    /// <summary>
    /// Downloads a catalog entry's weights (<see cref="EncoderDefinition.Url"/>) — and a text encoder's tokenizer model
    /// (<see cref="EncoderDefinition.TokenizerUrl"/>) — and installs them, refusing a file whose SHA-256 is not the one
    /// the entry pins. Downloads land beside the encoders, so a partial one never takes the place of an installed
    /// encoder, and are removed whatever happens.
    /// </summary>
    /// <param name="definition">The entry — with a URL and a SHA-256 (<see cref="EncoderCatalog"/>).</param>
    /// <param name="http">The client to download with.</param>
    /// <param name="progress">Bytes of the weights downloaded so far, and the total when the server says it.</param>
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
        var text = definition.Kind == EncoderKind.Text;
        if (text && (definition.TokenizerUrl is not { Length: > 0 } || definition.TokenizerSha256 is not { Length: > 0 }))
            throw new ArgumentException($"Encoder '{definition.Id}' reads text but pins no tokenizer download (URL and SHA-256); add it from files instead.", nameof(definition));

        Directory.CreateDirectory(Root);
        var download = Path.Combine(Root, definition.Id + ".download");
        var tokenizerDownload = Path.Combine(Root, definition.Id + ".tokenizer.download");
        try
        {
            await DownloadAsync(http, url, pinned, download, definition.Id, progress, cancellationToken).ConfigureAwait(false);
            if (text)
                await DownloadAsync(http, definition.TokenizerUrl!, definition.TokenizerSha256!, tokenizerDownload, definition.Id, null, cancellationToken).ConfigureAwait(false);
            return Add(definition, download, text ? tokenizerDownload : null);
        }
        finally
        {
            foreach (var file in new[] { download, tokenizerDownload }.Where(File.Exists))
                File.Delete(file);
        }
    }

    // Downloads url to target, refusing a file whose SHA-256 is not the pinned one.
    private static async Task DownloadAsync(
        HttpClient http, string url, string pinned, string target, string id,
        IProgress<(long Downloaded, long? Total)>? progress, CancellationToken cancellationToken)
    {
        using (var response = await http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false))
        {
            response.EnsureSuccessStatusCode();
            var total = response.Content.Headers.ContentLength;
            await using var source = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
            await using var file = new FileStream(target, FileMode.Create, FileAccess.Write, FileShare.None, 81920, useAsync: true);
            var buffer = new byte[81920];
            long downloaded = 0;
            int read;
            while ((read = await source.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
            {
                await file.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
                downloaded += read;
                progress?.Report((downloaded, total));
            }
        }

        var sha = Sha256Of(target);
        if (!sha.Equals(pinned, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException($"The download of encoder '{id}' ({url}) has SHA-256 {sha}, not the pinned {pinned}; nothing was installed.");
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
        var tokenizerPath = Path.Combine(dir, TokenizerFileName);
        if (!File.Exists(definitionPath) || !File.Exists(modelPath))
            return null;
        try
        {
            var definition = WithCatalogPatches(ReadDefinition(definitionPath));
            if (definition.Id != id)
                return null;
            if (definition.Kind != EncoderKind.Text)
                return new InstalledEncoder(definition, modelPath);
            return File.Exists(tokenizerPath) ? new InstalledEncoder(definition, modelPath, tokenizerPath) : null;
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

    // The catalog's command when the catalog names this encoder (or, with no name, any of the kind); adding files otherwise.
    private static string HowToInstall(string? id, EncoderKind kind)
    {
        var add = kind == EncoderKind.Text
            ? "mloop encoder add <encoder.json> --model <model.onnx> --tokenizer <tokenizer.model>"
            : "mloop encoder add <encoder.json> --model <model.onnx>";
        if (EncoderCatalog.Find(id) is { } entry)
            return $"Install it with: mloop encoder install {entry.Id}";
        return id is null
            ? (EncoderCatalog.InstallCommand(kind) is { } install ? $"Install one with: {install}, or add your own: {add}" : $"Add one with: {add}")
            : $"Add it with: {add}";
    }

    private static string Noun(EncoderKind kind) => kind == EncoderKind.Text ? "text" : "images";

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
