using System.Text;
using FilePrepperEncoding = FilePrepper.Utils.EncodingDetector;

namespace MLoop.Core.Data;

/// <summary>
/// Detects text file encoding with focus on Korean character encodings
/// (UTF-8, CP949/EUC-KR) and converts to UTF-8-with-BOM for ML.NET's TextLoader.
/// <para>
/// The byte-pattern detection itself is delegated to <see cref="FilePrepper.Utils.EncodingDetector"/>,
/// the single authority for this knowledge across the iyulab data stack — MLoop.Core already
/// depends on FilePrepper (the file-prep layer), so re-implementing CP949 detection here was
/// duplication that could drift. This type keeps MLoop's convenience surface (a typed
/// <see cref="DetectionResult"/> and the BOM-writing <see cref="ConvertToUtf8WithBom"/> policy that
/// ML.NET's TextLoader expects) as a thin adapter over that authority.
/// </para>
/// </summary>
public static class EncodingDetector
{
    /// <summary>
    /// Detection result with encoding information.
    /// </summary>
    public sealed record DetectionResult
    {
        public Encoding Encoding { get; init; } = Encoding.UTF8;
        public string EncodingName { get; init; } = "UTF-8";
        public bool HasBom { get; init; }
        public bool WasConverted { get; init; }
    }

    /// <summary>
    /// Detects the encoding of a file, delegating byte-pattern analysis to the FilePrepper authority.
    /// </summary>
    /// <param name="filePath">Path to the file to analyze.</param>
    /// <returns>Detection result with encoding information.</returns>
    public static DetectionResult DetectEncoding(string filePath)
    {
        var encoding = FilePrepperEncoding.DetectEncoding(filePath);
        return new DetectionResult
        {
            Encoding = encoding,
            EncodingName = DescribeEncoding(encoding),
            HasBom = HasByteOrderMark(filePath)
        };
    }

    /// <summary>
    /// Converts a file to UTF-8 with BOM if needed (ML.NET's TextLoader reads UTF-8 reliably only
    /// with the BOM present). Returns the original path untouched when it is already UTF-8.
    /// </summary>
    /// <param name="filePath">Path to the file.</param>
    /// <returns>Path to the UTF-8 file (a temp file when conversion was needed) and the detection.</returns>
    public static (string convertedPath, DetectionResult detectionResult) ConvertToUtf8WithBom(string filePath)
    {
        var detection = DetectEncoding(filePath);

        // If already UTF-8 (with or without BOM), return original — re-encoding a valid UTF-8 file
        // through an intermediate temp file can cause subtle issues.
        if (detection.Encoding.CodePage == Encoding.UTF8.CodePage)
        {
            return (filePath, detection);
        }

        // Read with detected encoding, write with UTF-8 BOM.
        var content = File.ReadAllText(filePath, detection.Encoding);
        var tempFile = Path.GetTempFileName();
        File.WriteAllText(tempFile, content, new UTF8Encoding(encoderShouldEmitUTF8Identifier: true));

        return (tempFile, detection with { WasConverted = true });
    }

    /// <summary>Human-readable name matching MLoop's historical labels (consumers log this).</summary>
    private static string DescribeEncoding(Encoding encoding) => encoding.CodePage switch
    {
        65001 => "UTF-8",                    // Encoding.UTF8
        949 => "CP949",                      // Korean (superset of EUC-KR)
        1200 => "UTF-16 LE",                 // Encoding.Unicode
        1201 => "UTF-16 BE",                 // Encoding.BigEndianUnicode
        _ => encoding.WebName
    };

    /// <summary>True when the file opens with a UTF-8 or UTF-16 (LE/BE) byte-order mark.</summary>
    private static bool HasByteOrderMark(string filePath)
    {
        Span<byte> head = stackalloc byte[3];
        using var fs = File.OpenRead(filePath);
        var n = fs.Read(head);
        if (n >= 3 && head[0] == 0xEF && head[1] == 0xBB && head[2] == 0xBF) return true; // UTF-8
        if (n >= 2 && head[0] == 0xFF && head[1] == 0xFE) return true;                    // UTF-16 LE
        if (n >= 2 && head[0] == 0xFE && head[1] == 0xFF) return true;                    // UTF-16 BE
        return false;
    }
}
