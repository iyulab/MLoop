using System.Globalization;
using System.Text;
using MLoop.CLI;

namespace MLoop.Tests.Commands;

/// <summary>
/// A multiclass label of class ids loads as one type in every load of a run — both partitions of the
/// split, <c>evaluate</c> and <c>predict</c> — however the rows of each file are ordered.
/// </summary>
/// <remarks>
/// Type inference reads a file's first ten thousand rows. In a file sorted by its label, the training
/// partition's first rows held only class <c>0</c> (read as yes/no, then loaded as a string) while the
/// test partition's held <c>0</c>, <c>1</c> and <c>2</c> (read as a number); the model, fitted on the
/// string, refused the test partition at the end of a full training run — <c>Could not apply a map over
/// type 'String' to column 'class' since it has type 'Single'</c>. Measured on a 35,000-row manufacturing
/// set sorted by class.
/// </remarks>
public class MulticlassLabelKindTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace =
        Path.Combine(Path.GetTempPath(), "mloop-label-kind-" + Guid.NewGuid().ToString("N"));

    public MulticlassLabelKindTests()
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

    private static void WriteSorted(string path, params (int Class, int Rows)[] blocks)
    {
        var random = new Random(7);
        var csv = new StringBuilder("x,y,class\n");
        foreach (var (cls, rows) in blocks)
        {
            for (var i = 0; i < rows; i++)
            {
                var x = cls * 2 + random.NextDouble();
                var y = random.NextDouble();
                csv.Append(string.Create(CultureInfo.InvariantCulture, $"{x:F4},{y:F4},{cls}\n"));
            }
        }
        File.WriteAllText(path, csv.ToString());
    }

    [Fact]
    public async Task AFileSortedByItsClassIds_Trains_Evaluates_AndPredicts()
    {
        var (initExit, initOutput, _) = await CliRunner.RunAsync("init", ".", "--task", "multiclass-classification", "--label", "class");
        Assert.True(initExit == 0, initOutput);

        // Enough leading zeros that the training partition's first ten thousand rows are all class 0,
        // while the test partition (a fifth) reaches class 2 well inside its own.
        WriteSorted(Path.Combine("datasets", "train.csv"), (0, 15_000), (1, 3_000), (2, 3_000));
        WriteSorted(Path.Combine("datasets", "test.csv"), (2, 50), (1, 50), (0, 50));

        var (trainExit, trainOutput, _) = await CliRunner.RunAsync(
            "train", "datasets/train.csv", "--label", "class", "--time", "60", "--max-trials", "10");
        Assert.True(trainExit == 0, $"train exited {trainExit}:\n{trainOutput}");

        var (evaluateExit, evaluateOutput, _) = await CliRunner.RunAsync("evaluate", "exp-001", "datasets/test.csv");
        Assert.True(evaluateExit == 0, $"evaluate exited {evaluateExit}:\n{evaluateOutput}");

        var (predictExit, predictOutput, _) = await CliRunner.RunAsync("predict", "datasets/test.csv");
        Assert.True(predictExit == 0, $"predict exited {predictExit}:\n{predictOutput}");
    }
}
