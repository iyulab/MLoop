using System.Globalization;
using System.Text;
using System.Text.Json;

namespace MLoop.Tests.Commands;

/// <summary>
/// A file saved from a dataframe with its index (the first column has an empty name, or "Unnamed: 0") is trained with that column
/// removed - and a model trained on it has to be usable. The input schema recorded the column as a feature, a column named "" could not
/// be given to the model's schema, and every prediction and every answer profile of such a model failed with "Name cannot be null or
/// white space" (a warning at the end of training said the answers could not be profiled; the failure itself came at the first predict).
/// </summary>
[Collection("FileSystem")]
public sealed class DataframeIndexColumnTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace = Path.Combine(Path.GetTempPath(), "mloop-dfindex-" + Guid.NewGuid().ToString("N"));

    public DataframeIndexColumnTests()
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
    [InlineData("")]
    [InlineData("Unnamed: 0")]
    public async Task AModelTrainedOnAFileWithAnIndexColumn_PredictsAndEvaluates_AndNeverNamesTheIndexColumn(string indexName)
    {
        Assert.Equal(0, (await CliRunner.RunAsync("init", "p", "--task", "regression")).ExitCode);
        Directory.SetCurrentDirectory(Path.Combine(_workspace, "p"));
        var rng = new Random(13);
        var text = new StringBuilder($"{indexName},size,rooms,price\n");
        for (var i = 0; i < 300; i++)
        {
            var size = rng.Next(30, 200);
            var rooms = rng.Next(1, 6);
            text.Append(string.Create(CultureInfo.InvariantCulture, $"{i},{size},{rooms},{size * 3.5 + rooms * 10 + rng.NextDouble() * 5:F2}\n"));
        }
        await File.WriteAllTextAsync("datasets/train.csv", text.ToString());

        var train = await CliRunner.RunAsync("train", "datasets/train.csv", "--label", "price", "--time", "20", "--max-trials", "6");
        var said = train.Stdout + train.Stderr;
        Assert.Equal(0, train.ExitCode);
        Assert.DoesNotContain("Could not profile", said, StringComparison.Ordinal);       // the answer profile was made
        Assert.DoesNotContain("Possible ID/index column", said, StringComparison.Ordinal);  // an index column the run removes is not "read as a feature"

        var predict = await CliRunner.RunAsync("predict", "datasets/train.csv", "--json");
        Assert.True(predict.ExitCode == 0, predict.Stdout + predict.Stderr);
        Assert.Equal(300, JsonDocument.Parse(predict.Stdout.Substring(predict.Stdout.IndexOf('{'))).RootElement.GetProperty("count").GetInt32());

        var evaluate = await CliRunner.RunAsync("evaluate", "datasets/train.csv", "--json");
        Assert.True(evaluate.ExitCode == 0, evaluate.Stdout + evaluate.Stderr);
    }
}
