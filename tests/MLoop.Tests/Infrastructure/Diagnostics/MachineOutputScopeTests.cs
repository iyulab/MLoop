using MLoop.CLI.Infrastructure.Diagnostics;
using Spectre.Console;

namespace MLoop.Tests.Infrastructure.Diagnostics;

/// <summary>
/// The scope that makes "<c>--json</c> ⇒ stdout carries only the event stream" true by construction
/// rather than by every call site remembering. <c>mloop train</c> renders through Spectre in dozens
/// of places and narrates through <c>Console.WriteLine</c> from three assemblies; one missed line
/// would make the stream unparseable for the consumer it exists for.
/// </summary>
[Collection("FileSystem")]
public class MachineOutputScopeTests
{
    [Fact]
    public void While_active_plain_console_writes_reach_nobody()
    {
        var captured = new StringWriter();
        var original = Console.Out;
        Console.SetOut(captured);
        try
        {
            using (new MachineOutputScope())
            {
                Console.WriteLine("narration that would corrupt the stream");
            }

            Assert.Equal(string.Empty, captured.ToString());
        }
        finally
        {
            Console.SetOut(original);
        }
    }

    [Fact]
    public void While_active_spectre_rendering_reaches_nobody()
    {
        var captured = new StringWriter();
        var originalOut = Console.Out;
        var originalConsole = AnsiConsole.Console;
        Console.SetOut(captured);
        AnsiConsole.Console = AnsiConsole.Create(new AnsiConsoleSettings { Out = new AnsiConsoleOutput(captured) });
        try
        {
            using (new MachineOutputScope())
            {
                AnsiConsole.MarkupLine("[green]a table nobody asked for[/]");
            }

            Assert.Equal(string.Empty, captured.ToString());
        }
        finally
        {
            AnsiConsole.Console = originalConsole;
            Console.SetOut(originalOut);
        }
    }

    [Fact]
    public void The_scope_hands_back_the_real_stdout_and_restores_it_on_dispose()
    {
        var captured = new StringWriter();
        var original = Console.Out;
        Console.SetOut(captured);
        try
        {
            using (var scope = new MachineOutputScope())
            {
                scope.Stdout.WriteLine("{\"event\":\"trial\"}");
            }

            Assert.Contains("{\"event\":\"trial\"}", captured.ToString());

            Console.WriteLine("after dispose");
            Assert.Contains("after dispose", captured.ToString());
        }
        finally
        {
            Console.SetOut(original);
        }
    }

    [Fact]
    public void A_failure_written_to_stderr_is_reported_to_the_event_stream_without_its_markup()
    {
        var original = Console.Out;
        Console.SetOut(new StringWriter());
        try
        {
            var reported = new List<string>();
            using (var scope = new MachineOutputScope())
            {
                scope.ErrorSink = reported.Add;
                ErrorConsole.Error("Training failed for model '[cyan]spike[/]'");
            }

            Assert.Equal("Training failed for model 'spike'", Assert.Single(reported));
        }
        finally
        {
            Console.SetOut(original);
        }
    }

    [Fact]
    public void Outside_the_scope_reporting_an_error_is_a_no_op()
    {
        // Every other command still calls the same stderr sink; it must not require a scope.
        MachineOutputScope.ReportError("no scope is active");
        Assert.Null(MachineOutputScope.Current);
    }

    [Fact]
    public void A_warning_raised_through_the_seam_is_reported_to_the_event_stream_without_its_markup()
    {
        var original = Console.Out;
        Console.SetOut(new StringWriter());
        try
        {
            var reported = new List<string>();
            using (var scope = new MachineOutputScope())
            {
                scope.WarningSink = reported.Add;
                WarningConsole.Warn("Class imbalance detected (ratio [cyan]19.0[/]:1)");
            }

            Assert.Equal("Class imbalance detected (ratio 19.0:1)", Assert.Single(reported));
        }
        finally
        {
            Console.SetOut(original);
        }
    }

    [Fact]
    public void Outside_machine_mode_a_warning_is_a_rendered_line_not_an_event()
    {
        var captured = new StringWriter();
        var originalConsole = AnsiConsole.Console;
        AnsiConsole.Console = AnsiConsole.Create(new AnsiConsoleSettings { Out = new AnsiConsoleOutput(captured) });
        try
        {
            WarningConsole.Warn("something recoverable happened");

            Assert.Contains("Warning:", captured.ToString());
            Assert.Contains("something recoverable happened", captured.ToString());
            Assert.Null(MachineOutputScope.Current);
        }
        finally
        {
            AnsiConsole.Console = originalConsole;
        }
    }
}
