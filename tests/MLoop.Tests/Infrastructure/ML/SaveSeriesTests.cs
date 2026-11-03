using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Models;
using MLoop.Core.Storage;

namespace MLoop.Tests.Infrastructure.ML;

/// <summary>
/// A forecasting model forecasts the points after its series, so the series is kept beside it: the file it was read from may be
/// moved, edited, or never reach the machine that serves the model.
/// </summary>
public sealed class SaveSeriesTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("mloop-series-").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { }
        GC.SuppressFinalize(this);
    }

    private string Write(params string[] lines)
    {
        var path = Path.Combine(_dir, $"{Guid.NewGuid():N}.csv");
        File.WriteAllLines(path, lines);
        return path;
    }

    private string Saved => Path.Combine(_dir, "exp", ExperimentLayout.SeriesFileName);

    private void Save(string data, string task = "forecasting", string label = "value")
    {
        Directory.CreateDirectory(Path.Combine(_dir, "exp"));
        TrainingEngine.SaveSeries(Path.Combine(_dir, "exp"), data, new TrainingConfig { ModelName = "m", DataFile = data, LabelColumn = label, Task = task });
    }

    [Fact]
    public void TheLabelColumn_IsKeptAlone_WhateverElseTheFileHolds()
    {
        Save(Write("date,value,note", "2026-01-01,10.5,a", "2026-01-02,11,b", "2026-01-03,12.25,\"c,d\""));

        Assert.Equal(["value", "10.5", "11", "12.25"], File.ReadAllLines(Saved).Select(l => l.TrimStart('﻿')));
    }

    [Fact]
    public void AOneColumnSeries_IsKeptAsItIs()
    {
        Save(Write("value", "1", "2", "3"));

        Assert.Equal(["value", "1", "2", "3"], File.ReadAllLines(Saved).Select(l => l.TrimStart('﻿')));
    }

    [Fact]
    public void AnotherTask_KeepsNothing()
    {
        Save(Write("value", "1", "2"), task: "regression");

        Assert.False(File.Exists(Saved));
    }

    [Fact]
    public void ALabelThatIsNotInTheFile_KeepsNothing()
    {
        Save(Write("a,b", "1,2"), label: "value");

        Assert.False(File.Exists(Saved));
    }
}
