using MLoop.CLI;
using MLoop.CLI.Commands;
using Spectre.Console;

namespace MLoop.Tests.Commands;

/// <summary>
/// What <c>mloop init</c> writes, <c>mloop validate</c> accepts — for every task type init offers.
/// </summary>
/// <remarks>
/// <para>
/// These two commands each carry their own list of the fifteen task types, and the configuration one
/// writes is the configuration the other judges. Nothing checked that the two agree: a task added to
/// init and not to validate would scaffold a project that the product's own validator rejects, and
/// the first person to find out would be a user one command into their first project.
/// </para>
/// <para>
/// Driven through <see cref="Program.ExecuteAsync"/> — the dispatch <c>Main</c> uses — rather than a
/// hand-rolled parse, so what is exercised is the command line a user actually types.
/// </para>
/// </remarks>
public class InitValidateRoundTripTests : IDisposable
{
    private readonly string _originalDirectory = Directory.GetCurrentDirectory();
    private readonly string _workspace =
        Path.Combine(Path.GetTempPath(), "mloop-init-validate-" + Guid.NewGuid().ToString("N"));

    public InitValidateRoundTripTests()
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

    /// <summary>
    /// The task types <c>init</c> accepts — read from the command itself, not copied. A third copy
    /// of this list would make the check green for exactly the task it stopped knowing about.
    /// </summary>
    public static TheoryData<string> TaskTypes => [.. InitCommand.ValidTasks];

    [Theory]
    [MemberData(nameof(TaskTypes))]
    public async Task AFreshProjectPassesValidation(string task)
    {
        var project = "p" + task.Replace("-", "", StringComparison.Ordinal);

        var (initExit, initOutput) = await RunAsync("init", project, "--task", task);
        Assert.True(initExit == 0, $"init --task {task} exited {initExit}: {initOutput}");

        Directory.SetCurrentDirectory(Path.Combine(_workspace, project));
        var (validateExit, validateOutput) = await RunAsync("validate");

        Assert.True(
            validateExit == 0,
            $"a project scaffolded with --task {task} does not pass the product's own validator "
            + $"(exit {validateExit}):{Environment.NewLine}{validateOutput}");
    }

    [Fact]
    public async Task TheCheckWouldNoticeATaskValidateDoesNotKnow()
    {
        // Negative control: the round trip has to be able to fail. A task init rejects never reaches
        // validate, and a config naming a task validate rejects must be reported — otherwise a green
        // run above would say nothing about whether the two lists agree.
        var (initExit, _) = await RunAsync("init", "pbogus", "--task", "not-a-real-task");
        Assert.Equal(1, initExit);

        Directory.CreateDirectory(Path.Combine(_workspace, "pmanual", ".mloop"));
        await File.WriteAllTextAsync(
            Path.Combine(_workspace, "pmanual", "mloop.yaml"),
            """
            project: pmanual
            models:
              default:
                task: not-a-real-task
                label: target
            """);

        Directory.SetCurrentDirectory(Path.Combine(_workspace, "pmanual"));
        var (validateExit, _) = await RunAsync("validate");
        Assert.Equal(1, validateExit);
    }

    // Console.SetOut alone does not reach Spectre's ambient renderer, so both are rebound — the same
    // pairing every command-driving test in this assembly uses.
    private static async Task<(int ExitCode, string Output)> RunAsync(params string[] args)
    {
        var originalOut = Console.Out;
        var originalAnsiConsole = AnsiConsole.Console;
        var buffer = new StringWriter();
        try
        {
            Console.SetOut(buffer);
            AnsiConsole.Console = AnsiConsole.Create(new AnsiConsoleSettings
            {
                Out = new AnsiConsoleOutput(buffer)
            });

            return (await Program.ExecuteAsync(Program.BuildRootCommand(), args), buffer.ToString());
        }
        finally
        {
            Console.SetOut(originalOut);
            AnsiConsole.Console = originalAnsiConsole;
        }
    }
}
