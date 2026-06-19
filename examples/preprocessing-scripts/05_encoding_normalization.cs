using MLoop.Extensibility.Preprocessing;
using System;
using System.IO;
using System.Text;
using System.Threading.Tasks;

/// <summary>
/// CSV encoding normalization to UTF-8.
/// Detects and converts non-UTF-8 files to UTF-8 with BOM.
///
/// Usage: Place in .mloop/scripts/preprocess/05_encoding_normalization.cs
/// Useful for datasets with encoding issues (e.g., Latin-1, Windows-1252).
///
/// Contract: IPreprocessingScript.ExecuteAsync returns the path to the produced CSV. When no
/// conversion is needed the original input path is returned unchanged.
/// </summary>
public class EncodingNormalizationScript : IPreprocessingScript
{
    public async Task<string> ExecuteAsync(PreprocessContext ctx)
    {
        ctx.Logger.Info("🔤 Encoding Normalization: Converting to UTF-8");

        var inputPath = ctx.InputPath;
        var encoding = await DetectEncodingAsync(inputPath);
        ctx.Logger.Info($"  Detected encoding: {encoding.EncodingName}");

        if (encoding == Encoding.UTF8)
        {
            ctx.Logger.Info("  ✓ Already UTF-8, no conversion needed");
            return inputPath;
        }

        // Read with the detected encoding, write back as UTF-8 (with BOM).
        var outputPath = Path.Combine(ctx.OutputDirectory, "05_utf8.csv");
        var content = await File.ReadAllTextAsync(inputPath, encoding);
        await File.WriteAllTextAsync(outputPath, content, new UTF8Encoding(encoderShouldEmitUTF8Identifier: true));

        ctx.Logger.Info($"  ✓ Converted {encoding.EncodingName} → UTF-8");
        ctx.Logger.Info($"✅ Saved: {outputPath}");
        return outputPath;
    }

    private async Task<Encoding> DetectEncodingAsync(string filePath)
    {
        // Read first 4KB to detect encoding.
        var buffer = new byte[4096];
        int bytesRead;

        using (var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read))
        {
            bytesRead = await fs.ReadAsync(buffer, 0, buffer.Length);
        }

        // Check for UTF-8 BOM
        if (bytesRead >= 3 && buffer[0] == 0xEF && buffer[1] == 0xBB && buffer[2] == 0xBF)
        {
            return Encoding.UTF8;
        }

        // Check for UTF-16 LE BOM
        if (bytesRead >= 2 && buffer[0] == 0xFF && buffer[1] == 0xFE)
        {
            return Encoding.Unicode;
        }

        // Check for UTF-16 BE BOM
        if (bytesRead >= 2 && buffer[0] == 0xFE && buffer[1] == 0xFF)
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
            // If UTF-8 validation fails, assume Windows-1252 (common for CSV).
            // Note: requires Encoding.RegisterProvider(CodePagesEncodingProvider.Instance) at the
            // host level for code pages outside the default set.
            return Encoding.GetEncoding("Windows-1252");
        }
    }
}
