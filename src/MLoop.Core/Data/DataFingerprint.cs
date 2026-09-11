using System.Security.Cryptography;

namespace MLoop.Core.Data;

/// <summary>
/// The fingerprint of a training data file — what an experiment records so that "which data was
/// this model trained on" has a checkable answer rather than a path.
/// </summary>
/// <remarks>
/// <para>
/// A path names a location, not a content: the file at <c>datasets/train.csv</c> is edited,
/// regenerated and replaced, and an experiment that recorded only the path cannot say whether the
/// file that is there today is the one it saw. The fingerprint is the content's SHA-256, written
/// as <c>sha256:&lt;hex&gt;</c> so the algorithm travels with the value and a later change of
/// algorithm cannot be mistaken for a change of data.
/// </para>
/// <para>
/// It is taken over the file <i>as the user has it</i>, before MLoop's own encoding conversion or
/// splitting produces temporary copies — a hash of a temporary file would be reproducible only by
/// MLoop, and the point is that anyone holding the original can confirm it. Two files with the
/// same bytes have the same fingerprint regardless of name or location.
/// </para>
/// </remarks>
public static class DataFingerprint
{
    /// <summary>The algorithm marker every fingerprint starts with.</summary>
    public const string Prefix = "sha256:";

    /// <summary>
    /// The fingerprint of the file at <paramref name="path"/>, streamed so a large file is not
    /// read into memory. Throws the ordinary I/O exception when the file cannot be read.
    /// </summary>
    public static async Task<string> ComputeAsync(string path, CancellationToken cancellationToken = default)
    {
        await using var stream = new FileStream(
            path, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 1 << 16, useAsync: true);
        var hash = await SHA256.HashDataAsync(stream, cancellationToken).ConfigureAwait(false);
        return Prefix + Convert.ToHexStringLower(hash);
    }

    private static readonly System.Buffers.SearchValues<char> LowerHex =
        System.Buffers.SearchValues.Create("0123456789abcdef");

    /// <summary>Whether <paramref name="value"/> has the shape this type writes.</summary>
    /// <remarks>
    /// The digits are checked against the sixteen characters themselves, not a character range:
    /// a range from <c>'0'</c> to <c>'f'</c> also spans the upper-case letters and the punctuation
    /// between them, and a first draft of this method let <c>ABCDEF</c> through that way.
    /// </remarks>
    public static bool IsFingerprint(string? value) =>
        value is not null
        && value.StartsWith(Prefix, StringComparison.Ordinal)
        && value.Length == Prefix.Length + 64
        && !value.AsSpan(Prefix.Length).ContainsAnyExcept(LowerHex);
}
