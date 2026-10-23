using System.CommandLine;
using System.Text.Json;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.Core.Embedding;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// <c>mloop encoder</c> — the pretrained image encoders <c>mloop embed</c> and <c>POST /embed</c> run, kept outside any
/// project (<see cref="EncoderStore"/>).
/// </summary>
public static class EncoderCommand
{
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase };

    public static Command Create()
    {
        var command = new Command("encoder", "Manage the pretrained image encoders mloop embed runs");
        command.Subcommands.Add(CreateListCommand());
        command.Subcommands.Add(CreateAddCommand());
        command.Subcommands.Add(CreateRemoveCommand());
        return command;
    }

    private static Command CreateListCommand()
    {
        var jsonOption = new Option<bool>("--json") { Description = "Emit the encoder list as JSON to stdout instead of the table" };
        var command = new Command("list", "List installed encoders");
        command.Options.Add(jsonOption);

        command.SetAction(parseResult =>
        {
            var store = new EncoderStore();
            var encoders = store.List();
            var json = parseResult.GetValue(jsonOption);
            using var machineOutput = json ? new JsonOutputScope() : null;
            if (json)
            {
                Console.WriteLine(JsonSerializer.Serialize(new { encoders = encoders.Select(e => EncoderIdentity.Of(e.Definition)) }, Json));
                return 0;
            }

            if (encoders.Count == 0)
            {
                AnsiConsole.MarkupLine($"[dim]No encoder installed in {Markup.Escape(store.Root)}.[/]");
                AnsiConsole.MarkupLine("Add one with: [cyan]mloop encoder add <encoder.json> --model <model.onnx>[/]");
                return 0;
            }

            var table = new Table();
            table.AddColumn("ID");
            table.AddColumn("Version");
            table.AddColumn("Dimension");
            table.AddColumn("Input");
            table.AddColumn("SHA-256");
            foreach (var e in encoders)
            {
                var d = e.Definition;
                table.AddRow(Markup.Escape(d.Id), Markup.Escape(d.Version), d.Dimension.ToString(System.Globalization.CultureInfo.InvariantCulture),
                    $"{d.Preprocessing.CropSize}²", Markup.Escape(d.Sha256?[..12] ?? "-"));
            }
            AnsiConsole.Write(table);
            return 0;
        });
        return command;
    }

    private static Command CreateAddCommand()
    {
        var definitionArg = new Argument<string>("definition") { Description = "The encoder's definition (encoder.json)" };
        var modelOption = new Option<string>("--model") { Description = "The encoder's ONNX graph", Required = true };
        var command = new Command("add", "Install an encoder from its definition and ONNX graph");
        command.Arguments.Add(definitionArg);
        command.Options.Add(modelOption);

        command.SetAction(parseResult =>
        {
            var definitionPath = parseResult.GetValue(definitionArg)!;
            var modelPath = parseResult.GetValue(modelOption)!;
            try
            {
                var installed = new EncoderStore().Add(EncoderStore.ReadDefinition(definitionPath), modelPath);
                AnsiConsole.MarkupLine($"[green]✓[/] Encoder [cyan]{Markup.Escape(installed.Definition.Id)}[/] installed " +
                                       $"({installed.Definition.Dimension} dimensions, SHA-256 {installed.Definition.Sha256![..12]}…)");
                return 0;
            }
            catch (Exception ex) when (ex is IOException or JsonException or InvalidDataException or ArgumentException or InvalidOperationException or UnauthorizedAccessException)
            {
                ErrorConsole.Error(ex.Message);
                return 1;
            }
        });
        return command;
    }

    private static Command CreateRemoveCommand()
    {
        var idArg = new Argument<string>("encoder") { Description = "The encoder to remove" };
        var command = new Command("remove", "Remove an installed encoder");
        command.Arguments.Add(idArg);

        command.SetAction(parseResult =>
        {
            var id = parseResult.GetValue(idArg)!;
            if (new EncoderStore().Remove(id))
                AnsiConsole.MarkupLine($"[green]✓[/] Encoder {Markup.Escape(id)} removed");
            else
                AnsiConsole.MarkupLine($"[dim]Encoder {Markup.Escape(id)} is not installed[/]");
            return 0;
        });
        return command;
    }
}
