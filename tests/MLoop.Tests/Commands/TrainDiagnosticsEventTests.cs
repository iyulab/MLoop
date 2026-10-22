using System.Globalization;
using System.Text;
using System.Text.Json;
using MLoop.CLI;

namespace MLoop.Tests.Commands;

/// <summary>
/// The performance diagnostics' warnings reach a <c>--json</c> consumer as <c>warning</c> events.
/// </summary>
/// <remarks>
/// The diagnostics block is narration, which machine mode silences, so its warnings — a near-perfect
/// score that may be a leak among them — reached only a person at the terminal.
/// </remarks>
public class TrainDiagnosticsEventTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace =
        Path.Combine(Path.GetTempPath(), "mloop-diag-events-" + Guid.NewGuid().ToString("N"));

    public TrainDiagnosticsEventTests()
    {
        Directory.CreateDirectory(_workspace);
        Directory.SetCurrentDirectory(_workspace);
    }

    public void Dispose()
    {
        GC.SuppressFinalize(this);
        try { Directory.SetCurrentDirectory(_originalDirectory); }
        catch { try { Directory.SetCurrentDirectory(Path.GetTempPath()); } catch { /* nothing left to do */ } }

        if (Directory.Exists(_workspace))
        {
            try { Directory.Delete(_workspace, recursive: true); } catch { /* a locked temp dir is not a test failure */ }
        }
    }

    [Fact]
    public async Task ANearPerfectScore_ArrivesAsAWarningEvent()
    {
        var (initExit, initOutput, _) = await CliRunner.RunAsync("init", ".", "--task", "binary-classification", "--label", "label");
        Assert.True(initExit == 0, initOutput);

        // The label is the feature crossing a line: a leak in its plainest form, held out perfectly.
        var random = new Random(5);
        var csv = new StringBuilder("x,noise,label\n");
        for (var i = 0; i < 800; i++)
        {
            var x = random.NextDouble();
            csv.Append(string.Create(CultureInfo.InvariantCulture, $"{x:F4},{random.NextDouble():F4},{(x > 0.5 ? "yes" : "no")}\n"));
        }
        File.WriteAllText(Path.Combine("datasets", "train.csv"), csv.ToString());

        var (exit, stdout, _) = await CliRunner.RunAsync(
            "train", "datasets/train.csv", "--label", "label", "--time", "5", "--no-promote", "--json");
        Assert.True(exit == 0, stdout);

        var warnings = stdout.Split('\n')
            .Where(l => l.TrimStart().StartsWith('{'))
            .Select(l => JsonDocument.Parse(l).RootElement)
            .Where(e => e.TryGetProperty("event", out var kind) && kind.GetString() == "warning")
            .Select(e => e.GetProperty("message").GetString()!)
            .ToList();
        Assert.Contains(warnings, w => w.Contains("Near-perfect on held-out rows"));
    }
}
