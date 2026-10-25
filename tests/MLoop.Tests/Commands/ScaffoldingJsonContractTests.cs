using System.Text.Json;
using MLoop.CLI;
using Spectre.Console;

namespace MLoop.Tests.Commands;

/// <summary>
/// <c>init</c>, <c>feedback add</c> and <c>sample create</c> were the last commands that handed a
/// caller prose to parse — <c>mloop-mcp</c> passed their text through. With <c>--json</c> stdout is one
/// document at every exit, success or failure, and the narration goes to stderr.
/// </summary>
public class ScaffoldingJsonContractTests
{
    private static (int ExitCode, string Stdout, string Stderr) Run(string directory, params string[] args)
    {
        var originalOut = Console.Out;
        var originalError = Console.Error;
        var originalAnsiConsole = AnsiConsole.Console;
        var originalDirectory = Directory.GetCurrentDirectory();

        var stdout = new StringWriter();
        var stderr = new StringWriter();
        try
        {
            Directory.SetCurrentDirectory(directory);
            Console.SetOut(stdout);
            Console.SetError(stderr);
            AnsiConsole.Console = AnsiConsole.Create(new AnsiConsoleSettings { Out = new AnsiConsoleOutput(stderr) });

            var exitCode = Program.Execute(Program.BuildRootCommand(), args);
            return (exitCode, stdout.ToString(), stderr.ToString());
        }
        finally
        {
            Console.SetOut(originalOut);
            Console.SetError(originalError);
            AnsiConsole.Console = originalAnsiConsole;
            Directory.SetCurrentDirectory(originalDirectory);
        }
    }

    private static DirectoryInfo Scratch() => Directory.CreateTempSubdirectory("mloop-scaffold-json-");

    private static void Cleanup(DirectoryInfo scratch)
    {
        try { scratch.Delete(recursive: true); } catch (IOException) { }
    }

    [Fact]
    public void InitReportsTheCreatedProjectAsOneDocument()
    {
        var scratch = Scratch();
        try
        {
            var (exit, stdout, stderr) = Run(scratch.FullName, "init", "proj", "--task", "regression", "--label", "Price", "--json");

            Assert.Equal(0, exit);
            using var document = JsonDocument.Parse(stdout);
            var root = document.RootElement;
            Assert.Equal("proj", root.GetProperty("project").GetString());
            Assert.Equal("regression", root.GetProperty("task").GetString());
            Assert.Equal("Price", root.GetProperty("label").GetString());
            Assert.False(root.GetProperty("reinitialized").GetBoolean());
            Assert.True(Directory.Exists(Path.Combine(scratch.FullName, "proj", ".mloop")));
            Assert.Contains("created successfully", stderr);
        }
        finally { Cleanup(scratch); }
    }

    [Fact]
    public void InitOnAnExistingDirectoryFailsWithAnErrorDocument()
    {
        var scratch = Scratch();
        try
        {
            Directory.CreateDirectory(Path.Combine(scratch.FullName, "proj"));

            var (exit, stdout, stderr) = Run(scratch.FullName, "init", "proj", "--json");

            Assert.Equal(1, exit);
            using var document = JsonDocument.Parse(stdout);
            Assert.Contains("already exists", document.RootElement.GetProperty("error").GetString());
            Assert.False(string.IsNullOrWhiteSpace(stderr));
        }
        finally { Cleanup(scratch); }
    }

    [Fact]
    public void SampleFromACsvReportsWhatItWrote()
    {
        var scratch = Scratch();
        try
        {
            var csv = Path.Combine(scratch.FullName, "data.csv");
            File.WriteAllLines(csv, ["a,b", .. Enumerable.Range(0, 20).Select(i => $"{i},{i * 2}")]);

            var (exit, stdout, _) = Run(scratch.FullName, "sample", "create", "--from", csv, "--rows", "5", "--json");

            Assert.Equal(0, exit);
            using var document = JsonDocument.Parse(stdout);
            var root = document.RootElement;
            Assert.Equal("csv", root.GetProperty("source").GetString());
            Assert.Equal(5, root.GetProperty("sampled").GetInt32());
            Assert.Equal(20, root.GetProperty("totalRows").GetInt32());
            Assert.True(File.Exists(root.GetProperty("outputPath").GetString()));
        }
        finally { Cleanup(scratch); }
    }

    [Fact]
    public void SampleWithoutASourceFailsWithAnErrorDocument()
    {
        var scratch = Scratch();
        try
        {
            var (exit, stdout, _) = Run(scratch.FullName, "sample", "create", "--json");

            Assert.Equal(1, exit);
            using var document = JsonDocument.Parse(stdout);
            Assert.Contains("--from", document.RootElement.GetProperty("error").GetString());
        }
        finally { Cleanup(scratch); }
    }

    [Fact]
    public void FeedbackForAnUnknownPredictionFailsWithAnErrorDocument()
    {
        var scratch = Scratch();
        try
        {
            Assert.Equal(0, Run(scratch.FullName, "init", "proj").ExitCode);
            var project = Path.Combine(scratch.FullName, "proj");

            var (exit, stdout, _) = Run(project, "feedback", "add", "-p", "no-such-id", "-v", "1", "--json");

            Assert.Equal(1, exit);
            using var document = JsonDocument.Parse(stdout);
            Assert.Contains("not found", document.RootElement.GetProperty("error").GetString());
        }
        finally { Cleanup(scratch); }
    }
}
