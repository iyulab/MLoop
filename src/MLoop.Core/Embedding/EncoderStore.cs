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
                : (null, $"Encoder '{id.Trim()}' is not installed. Installed: {Names()}. Add one with: mloop encoder add <encoder.json> --model <model.onnx>");
        }
        var all = List();
        return all.Count switch
        {
            1 => (all[0], null),
            0 => (null, "No encoder is installed. Add one with: mloop encoder add <encoder.json> --model <model.onnx>"),
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
            var definition = ReadDefinition(definitionPath);
            return definition.Id == id ? new InstalledEncoder(definition, modelPath) : null;
        }
        catch (JsonException)
        {
            return null;
        }
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
