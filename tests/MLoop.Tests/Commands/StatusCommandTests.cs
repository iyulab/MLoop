using MLoop.CLI.Commands;

namespace MLoop.Tests.Commands;

public class StatusCommandTests : IDisposable
{
    private readonly string _tempDir;

    public StatusCommandTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "mloop-status-test-" + Guid.NewGuid());
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, recursive: true);
    }

    #region GetLatestPrediction

    [Fact]
    public void GetLatestPrediction_NoDirectory_ReturnsDash()
    {
        var nonExistent = Path.Combine(_tempDir, "nope");

        var result = StatusCommand.GetLatestPrediction(nonExistent, "default");

        Assert.Contains("-", result);
    }

    [Fact]
    public void GetLatestPrediction_EmptyDirectory_ReturnsDash()
    {
        var result = StatusCommand.GetLatestPrediction(_tempDir, "default");

        Assert.Contains("-", result);
    }

    [Fact]
    public void GetLatestPrediction_NoMatchingFiles_ReturnsDash()
    {
        File.WriteAllText(Path.Combine(_tempDir, "other-model-predictions-001.csv"), "data");

        var result = StatusCommand.GetLatestPrediction(_tempDir, "default");

        Assert.Contains("-", result);
    }

    [Fact]
    public void GetLatestPrediction_RecentFile_ShowsMinutesAgo()
    {
        var filePath = Path.Combine(_tempDir, "default-predictions-001.csv");
        File.WriteAllText(filePath, "data");
        File.SetLastWriteTimeUtc(filePath, DateTime.UtcNow.AddMinutes(-15));

        var result = StatusCommand.GetLatestPrediction(_tempDir, "default");

        Assert.Contains("15m ago", result);
        Assert.Contains("[green]", result);
    }

    [Fact]
    public void GetLatestPrediction_HoursOld_ShowsHoursAgo()
    {
        var filePath = Path.Combine(_tempDir, "default-predictions-001.csv");
        File.WriteAllText(filePath, "data");
        File.SetLastWriteTimeUtc(filePath, DateTime.UtcNow.AddHours(-5));

        var result = StatusCommand.GetLatestPrediction(_tempDir, "default");

        Assert.Contains("5h ago", result);
    }

    [Fact]
    public void GetLatestPrediction_DaysOld_ShowsDaysAgo()
    {
        var filePath = Path.Combine(_tempDir, "default-predictions-001.csv");
        File.WriteAllText(filePath, "data");
        File.SetLastWriteTimeUtc(filePath, DateTime.UtcNow.AddDays(-10));

        var result = StatusCommand.GetLatestPrediction(_tempDir, "default");

        Assert.Contains("10d ago", result);
        Assert.Contains("[yellow]", result);
    }

    /// <summary>
    /// An age past a month is still said as an age ("2mo ago"), not as a calendar date. This test
    /// used to assert the date, and the same test in `mloop list` asserted a date at a different
    /// cutoff (a week) — the two assertions were pinning a divergence rather than a contract, and
    /// one of the two rendered its date from UTC while the other converted to the reader's zone.
    /// </summary>
    [Fact]
    public void GetLatestPrediction_OlderThanAMonth_IsStillAnAge()
    {
        var filePath = Path.Combine(_tempDir, "default-predictions-001.csv");
        File.WriteAllText(filePath, "data");
        var oldDate = DateTime.UtcNow.AddDays(-60);
        File.SetLastWriteTimeUtc(filePath, oldDate);

        var result = StatusCommand.GetLatestPrediction(_tempDir, "default");

        Assert.Contains("2mo ago", result);
        Assert.DoesNotContain(oldDate.ToString("yyyy-MM-dd"), result);
        Assert.Contains("[grey]", result);
    }

    [Fact]
    public void GetLatestPrediction_MultipleFiles_UsesLatest()
    {
        var file1 = Path.Combine(_tempDir, "default-predictions-001.csv");
        var file2 = Path.Combine(_tempDir, "default-predictions-002.csv");

        File.WriteAllText(file1, "data");
        File.SetLastWriteTimeUtc(file1, DateTime.UtcNow.AddDays(-10));

        File.WriteAllText(file2, "data");
        File.SetLastWriteTimeUtc(file2, DateTime.UtcNow.AddMinutes(-5));

        var result = StatusCommand.GetLatestPrediction(_tempDir, "default");

        Assert.Contains("5m ago", result);
    }

    #endregion
}
