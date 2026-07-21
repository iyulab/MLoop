using MLoop.CLI.Commands;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.CLI.Infrastructure.FileSystem;
using MLoop.CLI.Infrastructure.ML;
using MLoop.Core.Models;

namespace MLoop.Tests.ML;

/// <summary>
/// Proves the warning wiring live, end to end: a real training run whose data trips a quality
/// finding must land a <c>warning</c> event on the machine-output stream. The unit tests around
/// <c>WarningConsole</c> cannot prove this — the original warning-event shape shipped with a unit
/// test calling the emitter directly, which passed while nothing in the product called it.
/// </summary>
[Collection("FileSystem")]
public class WarningEventWiringTests : IDisposable
{
    private readonly string _tempDir;

    public WarningEventWiringTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "mloop-warnwire-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(_tempDir, ".mloop"));
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, true);
    }

    [Fact]
    public async Task A_data_quality_finding_from_a_real_run_reaches_the_event_stream()
    {
        // 380:20 — a 19:1 imbalance, comfortably past the validator's 10:1 "high" threshold.
        var csv = Path.Combine(_tempDir, "imbalanced.csv");
        var lines = new List<string> { "age,income,label" };
        var rnd = new Random(3);
        for (int i = 0; i < 400; i++)
            lines.Add($"{rnd.Next(20, 70)},{rnd.Next(1000, 9000)},{(i < 20 ? 1 : 0)}");
        await File.WriteAllLinesAsync(csv, lines);

        var config = new TrainingConfig
        {
            ModelName = "warnwire",
            DataFile = csv,
            LabelColumn = "label",
            Task = "binary-classification",
            TimeLimitSeconds = 5
        };

        var stream = new StringWriter();
        var originalOut = Console.Out;
        Console.SetOut(stream);
        try
        {
            // Wired exactly the way TrainCommand wires it, so this exercises the product seam.
            using var scope = new MachineOutputScope();
            var emitter = new TrainJsonEmitter(scope.Stdout);
            scope.WarningSink = emitter.Warning;

            var fs = new FileSystemManager();
            var store = new ExperimentStore(fs, new ProjectDiscovery(fs), _tempDir);
            try
            {
                await new TrainingEngine(fs, store).TrainAsync(config, progress: null, CancellationToken.None);
            }
            catch (Exception)
            {
                // The quality warnings are raised before training; whether the 5-second search
                // then succeeds is irrelevant to what this test pins.
            }
        }
        finally
        {
            Console.SetOut(originalOut);
        }

        var warnings = stream.ToString()
            .Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries)
            .Select(l => System.Text.Json.JsonDocument.Parse(l).RootElement)
            .Where(e => e.GetProperty("event").GetString() == "warning")
            .ToList();

        Assert.NotEmpty(warnings);
        Assert.Contains(warnings, w =>
            w.GetProperty("message").GetString()!.Contains("imbalance", StringComparison.OrdinalIgnoreCase));
    }
}
