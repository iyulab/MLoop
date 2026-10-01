using Microsoft.ML;
using MLoop.Core.Data;

namespace MLoop.Core.Tests.Data;

/// <summary>
/// "This column looks like an index; drop it" is advice about features. A series model reads only its
/// series column, so a timestamp counter beside it is not a feature it could overfit to.
/// </summary>
public class IndexColumnAdviceTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("mloop-index-advice-").FullName;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private List<string> LoadAndListen(string task)
    {
        var path = Path.Combine(_dir, $"{task}.csv");
        File.WriteAllLines(path, ["tick,reading,value", .. Enumerable.Range(0, 40).Select(i => $"{1000 + i},{(i * 7) % 5}.5,{i % 3}.25")]);
        var heard = new List<string>();
        new CsvDataLoader(new MLContext(seed: 0), heard.Add).LoadData(path, "value", task);
        return heard;
    }

    [Fact]
    public void A_series_task_is_not_told_to_drop_an_index_it_never_reads() =>
        Assert.DoesNotContain(LoadAndListen("time-series-anomaly"), m => m.Contains("Possible ID/index", StringComparison.Ordinal));

    [Fact]
    public void A_task_that_reads_features_is_still_told() =>
        Assert.Contains(LoadAndListen("regression"), m => m.Contains("Possible ID/index", StringComparison.Ordinal));
}
