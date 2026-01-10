using System.Text;
using MLoop.Extensibility.Preprocessing;

/// <summary>
/// CSV encoding normalization to UTF-8.
/// Detects and converts non-UTF-8 files to UTF-8 with BOM.
///
/// Usage: Place in .mloop/scripts/preprocess/05_encoding_normalization.cs
/// Useful for datasets with encoding issues (e.g., Latin-1, Windows-1252).
/// </summary>
public class EncodingNormalizationScript : IPreprocessingScript
{
    public async Task<PreprocessingResult> ExecuteAsync(PreprocessingContext context)
    {
        context.Logger.Info("🔤 Encoding Normalization: Converting to UTF-8");

        var inputPath = context.InputPath;
        var outputPath = context.GetTempPath("utf8.csv");

        try
        {
            // Detect current encoding
            var encoding = await DetectEncodingAsync(inputPath);
            context.Logger.Info($"  Detected encoding: {encoding.EncodingName}");

            if (encoding == Encoding.UTF8)
            {
                context.Logger.Info("  ✓ Already UTF-8, no conversion needed");
                return new PreprocessingResult
                {
                    OutputPath = inputPath,  // No change needed
                    Success = true,
                    Message = "File already in UTF-8 encoding"
                };
            }

            // Read with detected encoding, write as UTF-8
            var content = await File.ReadAllTextAsync(inputPath, encoding);
            await File.WriteAllTextAsync(outputPath, content, new UTF8Encoding(encoderShouldEmitUTF8Identifier: true));

            context.Logger.Info($"  ✓ Converted {encoding.EncodingName} → UTF-8");
            context.Logger.Info($"  💾 Saved: {outputPath}");

            return new PreprocessingResult
            {
                OutputPath = outputPath,
                Success = true,
                Message = $"Encoding converted from {encoding.EncodingName} to UTF-8"
            };
        }
        catch (Exception ex)
        {
            context.Logger.Error($"❌ Encoding normalization failed: {ex.Message}");
            return new PreprocessingResult
            {
                OutputPath = inputPath,  // Return original on failure
                Success = false,
                Message = $"Encoding conversion failed: {ex.Message}"
            };
        }
    }

    private async Task<Encoding> DetectEncodingAsync(string filePath)
    {
        // Read first 4KB to detect encoding
        var buffer = new byte[4096];
        int bytesRead;

        using (var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read))
        {
            bytesRead = await fs.ReadAsync(buffer, 0, buffer.Length);
        }

        // Check for UTF-8 BOM
        if (bytesRead >= 3 &&
            buffer[0] == 0xEF &&
            buffer[1] == 0xBB &&
            buffer[2] == 0xBF)
        {
            return Encoding.UTF8;
        }

        // Check for UTF-16 LE BOM
        if (bytesRead >= 2 &&
            buffer[0] == 0xFF &&
            buffer[1] == 0xFE)
        {
            return Encoding.Unicode;
        }

        // Check for UTF-16 BE BOM
        if (bytesRead >= 2 &&
            buffer[0] == 0xFE &&
            buffer[1] == 0xFF)
        {
            return Encoding.BigEndianUnicode;
        }

        // Try to validate as UTF-8
        try
        {
            var decoder = Encoding.UTF8.GetDecoder();
            var chars = new char[bytesRead];
            decoder.Convert(buffer, 0, bytesRead, chars, 0, bytesRead, false, out _, out _, out _);
            return Encoding.UTF8;
        }
        catch
        {
            // If UTF-8 validation fails, assume Windows-1252 (common for CSV)
            return Encoding.GetEncoding("Windows-1252");
        }
    }
}
