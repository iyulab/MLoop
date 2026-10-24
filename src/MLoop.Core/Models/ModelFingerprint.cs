using System.Collections.Concurrent;
using System.Security.Cryptography;

namespace MLoop.Core.Models;

/// <summary>
/// The SHA-256 of a model file — which model answered, in a form unique beyond the project. An experiment id
/// (<c>exp-001</c>) names a model only inside one project's models folder; a caller that records answers from
/// several projects or machines, as a pretrained encoder's <c>sha256</c> is recorded, needs the content.
/// </summary>
/// <remarks>
/// Remembered per file while its size and last-write time stay the same — the model cache's own test of
/// whether a model changed — so a server answering many requests with one model reads it once. A model of
/// a couple of hundred megabytes takes about a second to hash.
/// </remarks>
public static class ModelFingerprint
{
    private static readonly ConcurrentDictionary<string, (DateTime WrittenUtc, long Length, string Sha256)> Known =
        new(StringComparer.Ordinal);

    /// <summary>The lowercase hex SHA-256 of the file at <paramref name="modelPath"/>.</summary>
    public static string Sha256(string modelPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);

        var file = new FileInfo(Path.GetFullPath(modelPath));
        var (written, length) = (file.LastWriteTimeUtc, file.Length);
        if (Known.TryGetValue(file.FullName, out var known) && known.WrittenUtc == written && known.Length == length)
            return known.Sha256;

        using var stream = file.OpenRead();
        var sha256 = Convert.ToHexStringLower(SHA256.HashData(stream));
        Known[file.FullName] = (written, length, sha256);
        return sha256;
    }
}
