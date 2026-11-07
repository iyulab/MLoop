using MLoop.Core.Data;

namespace MLoop.Core.Tests.Data;

/// <summary>
/// <see cref="CsvDataLoader.FeatureColumnsLeft"/> — the columns a model could still read once the label, the excluded
/// columns and the columns with another role are set aside. Empty is the signal that nothing can train.
/// </summary>
public sealed class FeatureColumnsLeftTests : IDisposable
{
    private readonly string _file = Path.Combine(Path.GetTempPath(), "mloop-left-" + Guid.NewGuid().ToString("N") + ".csv");

    public void Dispose()
    {
        try { File.Delete(_file); } catch { /* a locked temp file is not a test failure */ }
    }

    private string Table(string header)
    {
        File.WriteAllText(_file, header + "\n1,2,3,4\n");
        return _file;
    }

    [Fact]
    public void TheLabelAndTheExcluded_AreSetApart_WhateverTheirCaseOrPadding()
    {
        var left = CsvDataLoader.FeatureColumnsLeft(Table("Id, Content ,w,Label"), "label", ["id", "content"], []);

        Assert.Equal(["w"], left);
    }

    [Fact]
    public void ARoleThatIsNotAFeature_IsSetApartToo()
    {
        var left = CsvDataLoader.FeatureColumnsLeft(Table("Content,w,g,Label"), "Label", ["Content"], ["w", "g"]);

        Assert.Empty(left);
    }

    [Fact]
    public void NothingExcluded_LeavesEveryColumnButTheLabel()
    {
        var left = CsvDataLoader.FeatureColumnsLeft(Table("a,b,Label"), "Label", [], []);

        Assert.Equal(["a", "b"], left);
    }
}
