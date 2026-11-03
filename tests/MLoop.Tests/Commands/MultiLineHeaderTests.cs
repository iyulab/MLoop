using System.Globalization;
using System.Text;
using System.Text.Json;

namespace MLoop.Tests.Commands;

/// <summary>
/// A header cell broken over two lines (Alt+Enter in a spreadsheet, quoted) is one column name, flattened to "a b" when the file is
/// trained on. A file with such a header has to train, predict and evaluate - in UTF-8 and in CP949, the encoding of the Korean
/// exports where it is common. Training flattened the file by reading it as UTF-8 (Korean became replacement characters), and
/// <c>evaluate</c> read the header as lines, so every column after the break was "missing".
/// </summary>
[Collection("FileSystem")]
public sealed class MultiLineHeaderTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-mlheader-" + Guid.NewGuid().ToString("N"));

    public MultiLineHeaderTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); } catch { Directory.SetCurrentDirectory(Path.GetTempPath()); }
        try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
    }

    [Theory]
    [InlineData(65001)]
    [InlineData(949)]
    public async Task AFileWithAHeaderOverTwoLines_TrainsPredictsAndEvaluates(int codePage)
    {
        Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
        var encoding = codePage == 65001 ? new UTF8Encoding(false) : Encoding.GetEncoding(codePage);
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "regression")).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        var rng = new Random(21);
        var text = new StringBuilder("크기,\"허용오차\n(상한값)\",방수,가격\n");
        for (var i = 0; i < 300; i++)
        {
            var size = rng.Next(30, 200);
            var tolerance = rng.Next(1, 6);
            text.Append(string.Create(CultureInfo.InvariantCulture, $"{size},{tolerance},{rng.Next(0, 3)},{size * 3.5 + tolerance * 10 + rng.NextDouble() * 5:F2}\n"));
        }
        await File.WriteAllTextAsync("datasets/train.csv", text.ToString(), encoding);

        var train = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "가격", "--time", "20", "--max-trials", "6");
        Assert.True(train.ExitCode == 0, train.Stdout + train.Stderr);

        var predict = await CliRunner.RunAsync("predict", "datasets/train.csv", "--json");
        Assert.True(predict.ExitCode == 0, predict.Stdout + predict.Stderr);
        Assert.Equal(300, JsonDocument.Parse(predict.Stdout.Substring(predict.Stdout.IndexOf('{'))).RootElement.GetProperty("count").GetInt32());

        var evaluate = await CliRunner.RunAsync("evaluate", "datasets/train.csv", "--json");
        Assert.True(evaluate.ExitCode == 0, evaluate.Stdout + evaluate.Stderr);
    }
}
