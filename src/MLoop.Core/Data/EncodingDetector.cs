using System.Text;

namespace MLoop.Core.Data;

/// <summary>
/// Detects text file encoding with focus on Korean character encodings.
/// Supports UTF-8, CP949 (Korean Windows), EUC-KR (Korean legacy).
/// </summary>
public static class EncodingDetector
{
    /// <summary>
    /// Detection result with encoding and confidence information.
    /// </summary>
    public sealed record DetectionResult
    {
        public Encoding Encoding { get; init; } = Encoding.UTF8;
        public string EncodingName { get; init; } = "UTF-8";
        public bool HasBom { get; init; }
        public float Confidence { get; init; }
        public bool WasConverted { get; init; }
    }

    /// <summary>
    /// Detects the encoding of a file by analyzing byte patterns.
    /// </summary>
    /// <param name="filePath">Path to the file to analyze.</param>
    /// <returns>Detection result with encoding information.</returns>
    public static DetectionResult DetectEncoding(string filePath)
    {
        byte[] buffer = new byte[Math.Min(64 * 1024, new FileInfo(filePath).Length)]; // Read up to 64KB
        int bytesRead;

        using (var fs = File.OpenRead(filePath))
        {
            bytesRead = fs.Read(buffer, 0, buffer.Length);
        }

        if (bytesRead == 0)
        {
            return new DetectionResult { Encoding = Encoding.UTF8, EncodingName = "UTF-8", Confidence = 1.0f };
        }

        // Check for BOM (Byte Order Mark)
        if (bytesRead >= 3 && buffer[0] == 0xEF && buffer[1] == 0xBB && buffer[2] == 0xBF)
        {
            return new DetectionResult
            {
                Encoding = Encoding.UTF8,
                EncodingName = "UTF-8",
                HasBom = true,
                Confidence = 1.0f
            };
        }

        // Check for UTF-16 BOM
        if (bytesRead >= 2)
        {
            if (buffer[0] == 0xFF && buffer[1] == 0xFE)
            {
                return new DetectionResult
                {
                    Encoding = Encoding.Unicode,
                    EncodingName = "UTF-16 LE",
                    HasBom = true,
                    Confidence = 1.0f
                };
            }

            if (buffer[0] == 0xFE && buffer[1] == 0xFF)
            {
                return new DetectionResult
                {
                    Encoding = Encoding.BigEndianUnicode,
                    EncodingName = "UTF-16 BE",
                    HasBom = true,
                    Confidence = 1.0f
                };
            }
        }

        // No BOM - analyze byte patterns
        var span = new ReadOnlySpan<byte>(buffer, 0, bytesRead);

        // Check if valid UTF-8
        var (isValidUtf8, utf8Confidence) = CheckUtf8Validity(span);
        if (isValidUtf8 && utf8Confidence > 0.9f)
        {
            return new DetectionResult
            {
                Encoding = Encoding.UTF8,
                EncodingName = "UTF-8",
                HasBom = false,
                Confidence = utf8Confidence
            };
        }

        // Check for Korean encodings (CP949/EUC-KR)
        var koreanConfidence = CheckKoreanEncoding(span);
        if (koreanConfidence > 0.5f)
        {
            // Register the code page (needed for .NET Core)
            Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);

            return new DetectionResult
            {
                Encoding = Encoding.GetEncoding(949), // CP949 (superset of EUC-KR)
                EncodingName = "CP949",
                HasBom = false,
                Confidence = koreanConfidence
            };
        }

        // Fallback to UTF-8
        return new DetectionResult
        {
            Encoding = Encoding.UTF8,
            EncodingName = "UTF-8",
            HasBom = false,
            Confidence = 0.5f
        };
    }

    /// <summary>
    /// Converts a file to UTF-8 with BOM if needed.
    /// </summary>
    /// <param name="filePath">Path to the file.</param>
    /// <returns>Path to the UTF-8 file (may be temp file if conversion was needed).</returns>
    public static (string convertedPath, DetectionResult detectionResult) ConvertToUtf8WithBom(string filePath)
    {
        var detection = DetectEncoding(filePath);

        // If already UTF-8 with BOM, return original
        if (detection.HasBom && detection.Encoding.CodePage == Encoding.UTF8.CodePage)
        {
            return (filePath, detection);
        }

        // Read with detected encoding, write with UTF-8 BOM
        var content = File.ReadAllText(filePath, detection.Encoding);
        var tempFile = Path.GetTempFileName();
        File.WriteAllText(tempFile, content, new UTF8Encoding(encoderShouldEmitUTF8Identifier: true));

        return (tempFile, detection with { WasConverted = true });
    }

    /// <summary>
    /// Checks if byte sequence is valid UTF-8 and returns confidence score.
    /// </summary>
    private static (bool isValid, float confidence) CheckUtf8Validity(ReadOnlySpan<byte> data)
    {
        int validMultibyteSequences = 0;
        int invalidSequences = 0;
        int asciiCount = 0;
        int i = 0;

        while (i < data.Length)
        {
            byte b = data[i];

            if (b <= 0x7F)
            {
                // ASCII character
                asciiCount++;
                i++;
            }
            else if ((b & 0xE0) == 0xC0)
            {
                // 2-byte sequence
                if (i + 1 >= data.Length || (data[i + 1] & 0xC0) != 0x80)
                {
                    invalidSequences++;
                    i++;
                }
                else
                {
                    validMultibyteSequences++;
                    i += 2;
                }
            }
            else if ((b & 0xF0) == 0xE0)
            {
                // 3-byte sequence (common for Korean UTF-8)
                if (i + 2 >= data.Length ||
                    (data[i + 1] & 0xC0) != 0x80 ||
                    (data[i + 2] & 0xC0) != 0x80)
                {
                    invalidSequences++;
                    i++;
                }
                else
                {
                    validMultibyteSequences++;
                    i += 3;
                }
            }
            else if ((b & 0xF8) == 0xF0)
            {
                // 4-byte sequence
                if (i + 3 >= data.Length ||
                    (data[i + 1] & 0xC0) != 0x80 ||
                    (data[i + 2] & 0xC0) != 0x80 ||
                    (data[i + 3] & 0xC0) != 0x80)
                {
                    invalidSequences++;
                    i++;
                }
                else
                {
                    validMultibyteSequences++;
                    i += 4;
                }
            }
            else
            {
                // Invalid UTF-8 lead byte
                invalidSequences++;
                i++;
            }
        }

        bool isValid = invalidSequences == 0;
        float confidence = invalidSequences == 0
            ? (validMultibyteSequences > 0 ? 0.95f : 0.8f) // Higher confidence if we saw multibyte
            : 1.0f - Math.Min(1.0f, invalidSequences / (float)data.Length * 10);

        return (isValid, confidence);
    }

    /// <summary>
    /// Checks for CP949/EUC-KR Korean encoding patterns.
    /// CP949 uses 0x81-0xFE for lead bytes and 0x41-0xFE for trail bytes.
    /// </summary>
    private static float CheckKoreanEncoding(ReadOnlySpan<byte> data)
    {
        int koreanByteSequences = 0;
        int invalidSequences = 0;
        int asciiCount = 0;
        int i = 0;

        while (i < data.Length)
        {
            byte b = data[i];

            if (b <= 0x7F)
            {
                // ASCII
                asciiCount++;
                i++;
            }
            else if (b >= 0x81 && b <= 0xFE)
            {
                // Potential CP949 lead byte
                if (i + 1 < data.Length)
                {
                    byte trail = data[i + 1];
                    // CP949 trail byte: 0x41-0xFE (excluding 0x7F)
                    if ((trail >= 0x41 && trail <= 0x5A) || // A-Z
                        (trail >= 0x61 && trail <= 0x7A) || // a-z
                        (trail >= 0x81 && trail <= 0xFE))   // High bytes
                    {
                        koreanByteSequences++;
                        i += 2;
                    }
                    else
                    {
                        invalidSequences++;
                        i++;
                    }
                }
                else
                {
                    invalidSequences++;
                    i++;
                }
            }
            else
            {
                // Invalid for CP949
                invalidSequences++;
                i++;
            }
        }

        if (koreanByteSequences == 0)
        {
            return 0.0f;
        }

        // Calculate confidence based on ratio of valid Korean sequences
        float totalSequences = koreanByteSequences + invalidSequences;
        float confidence = koreanByteSequences / totalSequences;

        // Boost confidence if many Korean sequences found
        if (koreanByteSequences > 10)
        {
            confidence = Math.Min(1.0f, confidence + 0.1f);
        }

        return confidence;
    }
}
