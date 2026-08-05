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

    /// <summary>
    /// Which columns featurization drops has to arrive as an event, not as <c>[Warning]</c> prose on
    /// the info channel. It was the latter: the removal chain narrated it through the logger's
    /// <c>Info</c>, so a <c>--json</c> run — whose stdout carries only events — was never told, and a
    /// consumer had to re-derive the post-exclusion schema to know what was trained on.
    /// </summary>
    [Fact]
    public async Task Feature_exclusions_from_a_real_run_reach_the_event_stream_with_their_reason()
    {
        // `note` never varies, so it is excluded as constant; the two real features stay.
        var csv = Path.Combine(_tempDir, "constant-column.csv");
        var lines = new List<string> { "age,income,note,label" };
        var rnd = new Random(11);
        for (int i = 0; i < 400; i++)
        {
            var age = rnd.Next(20, 70);
            var income = rnd.Next(1000, 9000);
            lines.Add($"{age},{income},same,{(income + age * 50 > 6000 ? 1 : 0)}");
        }
        await File.WriteAllLinesAsync(csv, lines);

        var warnings = await CaptureWarningsAsync(new TrainingConfig
        {
            ModelName = "exclusionwire",
            DataFile = csv,
            LabelColumn = "label",
            Task = "binary-classification",
            TimeLimitSeconds = 5
        });

        var excluded = Assert.Single(warnings, w =>
            w.GetProperty("message").GetString()!.Contains("Excluded from features"));
        var message = excluded.GetProperty("message").GetString()!;
        Assert.Contains("note", message);
        // The reason travels with it: "excluded" alone does not tell a consumer whether the column
        // was constant, sparse, or a timestamp.
        Assert.Contains("constant", message, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// The gap the promotion gate leaves: judged on <c>f1_score</c>, a model can promote while its
    /// accuracy is below what a constant predictor scores, because the f1 floor knows nothing about
    /// the majority share. The number then reads as good and is not.
    /// </summary>
    [Fact]
    public async Task Accuracy_below_the_majority_share_is_warned_when_the_gate_judges_another_metric()
    {
        // 8 positives in 400 rows: a constant "0" predictor scores 0.98, which almost anything the
        // search produces will fail to beat on this signal-free label.
        var csv = Path.Combine(_tempDir, "majority-baseline.csv");
        var lines = new List<string> { "age,income,label" };
        var rnd = new Random(17);
        for (int i = 0; i < 400; i++)
            lines.Add($"{rnd.Next(20, 70)},{rnd.Next(1000, 9000)},{(i % 50 == 0 ? 1 : 0)}");
        await File.WriteAllLinesAsync(csv, lines);

        var warnings = await CaptureWarningsAsync(new TrainingConfig
        {
            ModelName = "baselinewire",
            DataFile = csv,
            LabelColumn = "label",
            Task = "binary-classification",
            Metric = "f1_score",   // not accuracy — so the gate's accuracy floor does not apply
            TimeLimitSeconds = 10
        });

        var warned = Assert.Single(warnings, w =>
            w.GetProperty("message").GetString()!.Contains("most common class"));
        // Both numbers travel with it: "below baseline" without the baseline cannot be acted on.
        var message = warned.GetProperty("message").GetString()!;
        Assert.Contains("Accuracy 0.", message);
        Assert.Matches(@"\d+\.\d%", message);
    }

    /// <summary>
    /// No warning event carries display markup. The seam strips Spectre tags on the way out, but
    /// only when the string parses as markup — one unescaped bracket and the raw text is emitted
    /// verbatim, tags and all. This is the guard for every call site at once: a machine-readable
    /// field is not where a colour belongs.
    /// </summary>
    [Fact]
    public async Task No_warning_event_carries_display_markup()
    {
        // A run that trips several distinct sources at once: missing labels, extreme imbalance,
        // a constant column, and an accuracy below the majority share.
        var csv = Path.Combine(_tempDir, "many-warnings.csv");
        var lines = new List<string> { "age,income,note,label" };
        var rnd = new Random(23);
        for (int i = 0; i < 300; i++)
        {
            var label = i is 7 or 33 or 150 ? "" : (i % 60 == 0 ? "1" : "0");
            lines.Add($"{rnd.Next(20, 70)},{rnd.Next(1000, 9000)},same,{label}");
        }
        await File.WriteAllLinesAsync(csv, lines);

        var warnings = await CaptureWarningsAsync(new TrainingConfig
        {
            ModelName = "markupwire",
            DataFile = csv,
            LabelColumn = "label",
            Task = "binary-classification",
            Metric = "f1_score",
            TimeLimitSeconds = 10
        });

        Assert.NotEmpty(warnings);
        Assert.All(warnings, w =>
            Assert.DoesNotMatch(@"\[/?[a-z]", w.GetProperty("message").GetString()!));
    }

    /// <summary>Runs a training pass with the machine-output wiring TrainCommand uses and returns its warning events.</summary>
    private async Task<List<System.Text.Json.JsonElement>> CaptureWarningsAsync(TrainingConfig config)
    {
        var stream = new StringWriter();
        var originalOut = Console.Out;
        Console.SetOut(stream);
        try
        {
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
                // Whether the short search then produces a model is irrelevant to the wiring pinned here.
            }
        }
        finally
        {
            Console.SetOut(originalOut);
        }

        return [.. stream.ToString()
            .Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries)
            .Select(l => System.Text.Json.JsonDocument.Parse(l).RootElement)
            .Where(e => e.GetProperty("event").GetString() == "warning")];
    }
}
