using System.Text;
using MLoop.Core.Data;

namespace MLoop.Core.Tests.Data;

public class EncodingDetectorTests : IDisposable
{
    private readonly List<string> _tempFiles = [];

    public void Dispose()
    {
        foreach (var file in _tempFiles)
        {
            try
            {
                if (File.Exists(file))
                    File.Delete(file);
            }
            catch
            {
                // Ignore cleanup errors
            }
        }
    }

    private string CreateTempFile(byte[] content)
    {
        var path = Path.GetTempFileName();
        File.WriteAllBytes(path, content);
        _tempFiles.Add(path);
        return path;
    }

    private string CreateTempFile(string content, Encoding encoding, bool includeBom = false)
    {
        var path = Path.GetTempFileName();
        // For UTF8, create encoding without BOM unless explicitly requested
        Encoding actualEncoding = encoding;
        if (encoding.CodePage == Encoding.UTF8.CodePage && !includeBom)
        {
            actualEncoding = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
        }
        File.WriteAllText(path, content, actualEncoding);
        _tempFiles.Add(path);
        return path;
    }

    #region UTF-8 Detection

    [Fact]
    public void DetectEncoding_Utf8WithBom_DetectsCorrectly()
    {
        // Create UTF-8 file with BOM
        var content = new byte[] { 0xEF, 0xBB, 0xBF, (byte)'H', (byte)'e', (byte)'l', (byte)'l', (byte)'o' };
        var path = CreateTempFile(content);

        var result = EncodingDetector.DetectEncoding(path);

        Assert.Equal("UTF-8", result.EncodingName);
        Assert.True(result.HasBom);
        Assert.Equal(1.0f, result.Confidence);
    }

    [Fact]
    public void DetectEncoding_Utf8NoBom_AsciiOnly_DetectsUtf8()
    {
        var path = CreateTempFile("Name,Value\nItem1,100\nItem2,200", Encoding.UTF8);

        var result = EncodingDetector.DetectEncoding(path);

        Assert.Equal("UTF-8", result.EncodingName);
        Assert.False(result.HasBom);
        Assert.True(result.Confidence >= 0.5f, $"Confidence {result.Confidence} should be >= 0.5 for ASCII content");
    }

    [Fact]
    public void DetectEncoding_Utf8NoBom_WithKorean_DetectsUtf8()
    {
        var path = CreateTempFile("설비명,공정명,값\n설비1,공정A,100", Encoding.UTF8);

        var result = EncodingDetector.DetectEncoding(path);

        Assert.Equal("UTF-8", result.EncodingName);
        Assert.True(result.Confidence >= 0.9f);
    }

    #endregion

    #region UTF-16 Detection

    [Fact]
    public void DetectEncoding_Utf16LE_WithBom_DetectsCorrectly()
    {
        // UTF-16 LE BOM: FF FE
        var content = new byte[] { 0xFF, 0xFE, (byte)'H', 0, (byte)'i', 0 };
        var path = CreateTempFile(content);

        var result = EncodingDetector.DetectEncoding(path);

        Assert.Equal("UTF-16 LE", result.EncodingName);
        Assert.True(result.HasBom);
    }

    [Fact]
    public void DetectEncoding_Utf16BE_WithBom_DetectsCorrectly()
    {
        // UTF-16 BE BOM: FE FF
        var content = new byte[] { 0xFE, 0xFF, 0, (byte)'H', 0, (byte)'i' };
        var path = CreateTempFile(content);

        var result = EncodingDetector.DetectEncoding(path);

        Assert.Equal("UTF-16 BE", result.EncodingName);
        Assert.True(result.HasBom);
    }

    #endregion

    #region Korean Encoding Detection

    [Fact]
    public void DetectEncoding_Cp949KoreanText_DetectsKorean()
    {
        // Register CP949 encoding
        Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
        var cp949 = Encoding.GetEncoding(949);

        // Korean text in CP949
        var koreanText = "설비명,설비번호,공정명\n열처리1,001,열처리공정";
        var bytes = cp949.GetBytes(koreanText);
        var path = CreateTempFile(bytes);

        var result = EncodingDetector.DetectEncoding(path);

        Assert.Equal("CP949", result.EncodingName);
        Assert.True(result.Confidence >= 0.5f);
    }

    #endregion

    #region ConvertToUtf8WithBom

    [Fact]
    public void ConvertToUtf8WithBom_AlreadyUtf8WithBom_ReturnsOriginalPath()
    {
        // Create file with UTF-8 BOM
        var content = new byte[] { 0xEF, 0xBB, 0xBF, (byte)'t', (byte)'e', (byte)'s', (byte)'t' };
        var path = CreateTempFile(content);

        var (convertedPath, detection) = EncodingDetector.ConvertToUtf8WithBom(path);

        Assert.Equal(path, convertedPath);
        Assert.False(detection.WasConverted);
    }

    [Fact]
    public void ConvertToUtf8WithBom_Utf8NoBom_ReturnsOriginalPath()
    {
        // UTF-8 without BOM is still valid UTF-8 — no conversion needed
        var path = CreateTempFile("test,data\n1,2", Encoding.UTF8);

        var (convertedPath, detection) = EncodingDetector.ConvertToUtf8WithBom(path);

        Assert.Equal(path, convertedPath);
        Assert.False(detection.WasConverted);
    }

    [Fact]
    public void ConvertToUtf8WithBom_Cp949_ConvertsToUtf8()
    {
        // Register CP949 encoding
        Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
        var cp949 = Encoding.GetEncoding(949);

        var koreanText = "설비명,값\n테스트,100";
        var bytes = cp949.GetBytes(koreanText);
        var path = CreateTempFile(bytes);

        var (convertedPath, detection) = EncodingDetector.ConvertToUtf8WithBom(path);
        _tempFiles.Add(convertedPath);

        Assert.True(detection.WasConverted);

        // Verify content is correct Korean
        var content = File.ReadAllText(convertedPath, Encoding.UTF8);
        Assert.Contains("설비명", content);
        Assert.Contains("테스트", content);
    }

    [Fact]
    public void DetectEncoding_Cp949Header_AsciiBody_DetectsKorean()
    {
        // Simulate real-world case: CP949 Korean headers + mostly numeric/ASCII body
        // This is common in KAMP datasets (Korean column names, numeric data)
        Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
        var cp949 = Encoding.GetEncoding(949);

        // Build a file with Korean headers and many ASCII data rows
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("uid,order_date,제품코드,제품명,총주문잔량");
        for (int i = 0; i < 1000; i++)
        {
            sb.AppendLine($"{i},2024-01-{(i % 28) + 1:D2},P{i:D4},Product{i},{i * 10}");
        }

        var bytes = cp949.GetBytes(sb.ToString());
        var path = CreateTempFile(bytes);

        var result = EncodingDetector.DetectEncoding(path);

        // Must detect as CP949, NOT UTF-8 — even though body is mostly ASCII
        Assert.Equal("CP949", result.EncodingName);
        Assert.True(result.Confidence >= 0.5f,
            $"Expected CP949 confidence >= 0.5, got {result.Confidence}");
    }

    [Fact]
    public void ConvertToUtf8WithBom_Cp949Header_AsciiBody_ConvertsCorrectly()
    {
        // Real-world case: CP949 headers preserved after conversion
        Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
        var cp949 = Encoding.GetEncoding(949);

        var sb = new System.Text.StringBuilder();
        sb.AppendLine("uid,제품등급,총주문잔량");
        for (int i = 0; i < 100; i++)
            sb.AppendLine($"{i},A등급,{i * 100}");

        var bytes = cp949.GetBytes(sb.ToString());
        var path = CreateTempFile(bytes);

        var (convertedPath, detection) = EncodingDetector.ConvertToUtf8WithBom(path);
        _tempFiles.Add(convertedPath);

        Assert.True(detection.WasConverted);

        var content = File.ReadAllText(convertedPath, Encoding.UTF8);
        Assert.Contains("제품등급", content);
        Assert.Contains("총주문잔량", content);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void DetectEncoding_EmptyFile_ReturnsUtf8Default()
    {
        var path = CreateTempFile(Array.Empty<byte>());

        var result = EncodingDetector.DetectEncoding(path);

        Assert.Equal("UTF-8", result.EncodingName);
        Assert.Equal(1.0f, result.Confidence);
    }

    [Fact]
    public void DetectEncoding_OnlyAscii_ReturnsUtf8()
    {
        var path = CreateTempFile("Hello,World\n123,456", Encoding.ASCII);

        var result = EncodingDetector.DetectEncoding(path);

        Assert.Equal("UTF-8", result.EncodingName);
    }

    #endregion
}
