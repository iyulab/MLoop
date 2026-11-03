using System.Text;
using MLoop.CLI.Infrastructure.ML;

namespace MLoop.Tests.Infrastructure.ML;

/// <summary>
/// Whether a label is a column of the file is judged on the header read in the file's own encoding: a Korean label name against a
/// CP949 file (the usual spreadsheet export) was refused as "not found", with the columns listed as replacement characters.
/// </summary>
public sealed class TrainingInputLabelEncodingTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "mloop-labelenc-" + Guid.NewGuid().ToString("N"));

    public TrainingInputLabelEncodingTests() => Directory.CreateDirectory(_directory);

    public void Dispose()
    {
        try { Directory.Delete(_directory, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    private string Write(string text, Encoding encoding)
    {
        var path = Path.Combine(_directory, "train.csv");
        File.WriteAllText(path, text, encoding);
        return path;
    }

    private const string Table = "uid,주문량,제품명\n1,10,가\n2,20,나\n3,30,다\n";

    [Fact]
    public void AKoreanLabel_IsAColumnOfACp949File()
    {
        Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
        var path = Write(Table, Encoding.GetEncoding(949));

        Assert.Null(TrainingInput.LabelColumnRefusal(path, "regression", "주문량"));
    }

    [Fact]
    public void AKoreanLabel_IsAColumnOfAUtf8File()
    {
        var path = Write(Table, new UTF8Encoding(encoderShouldEmitUTF8Identifier: true));

        Assert.Null(TrainingInput.LabelColumnRefusal(path, "regression", "주문량"));
    }

    [Fact]
    public void ALabelThatIsNoColumnOfACp949File_IsRefusedWithTheColumnsReadable()
    {
        Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
        var path = Write(Table, Encoding.GetEncoding(949));

        var refusal = TrainingInput.LabelColumnRefusal(path, "regression", "없는열");

        Assert.NotNull(refusal);
        Assert.Contains("Available columns: uid, 주문량, 제품명", refusal, StringComparison.Ordinal);
    }
}
