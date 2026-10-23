using System.CommandLine;
using System.Globalization;
using System.Text.Json;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.Core.Embedding;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// <c>mloop embed &lt;images…&gt;</c> — one vector per image from an installed pretrained encoder, with the encoder's
/// identity (id and weights' SHA-256) that a consumer keeps beside them. Needs no project.
/// </summary>
public static class EmbedCommand
{
    private static readonly JsonSerializerOptions Json = new() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };

    public static Command Create()
    {
        var imagesArg = new Argument<string[]>("images")
        {
            Description = $"Image files (PNG, JPEG, …), at most {EmbedderPool.MaxImages}",
            Arity = ArgumentArity.OneOrMore,
        };
        var encoderOption = new Option<string?>("--encoder") { Description = "The encoder to use; may be left out when only one is installed" };
        var jsonOption = new Option<bool>("--json") { Description = "Emit the encoder and every vector as JSON to stdout" };

        var command = new Command("embed", "Embed images as vectors with a pretrained encoder");
        command.Arguments.Add(imagesArg);
        command.Options.Add(encoderOption);
        command.Options.Add(jsonOption);

        command.SetAction(parseResult =>
        {
            var paths = parseResult.GetValue(imagesArg)!;
            var json = parseResult.GetValue(jsonOption);
            using var machineOutput = json ? new JsonOutputScope() : null;
            if (paths.FirstOrDefault(p => !File.Exists(p)) is { } missing)
            {
                ErrorConsole.Error($"Image not found: {missing}");
                return 1;
            }

            using var pool = new EmbedderPool(new EncoderStore());
            EmbeddingResult? result;
            string? refusal;
            try
            {
                (result, refusal) = pool.Embed(parseResult.GetValue(encoderOption), [.. paths.Select(File.ReadAllBytes)]);
            }
            catch (UnreadableImageException ex)
            {
                ErrorConsole.Error($"{paths[ex.Index]}: {ex.Message}");
                return 1;
            }
            if (result is null)
            {
                ErrorConsole.Error(refusal!);
                return 1;
            }

            if (json)
            {
                Console.WriteLine(JsonSerializer.Serialize(result, Json));
                return 0;
            }

            var e = result.Encoder;
            AnsiConsole.MarkupLine($"Encoder [cyan]{Markup.Escape(e.Id)}[/] {Markup.Escape(e.Version)} · {e.Dimension} dimensions · SHA-256 {Markup.Escape(e.Sha256?[..12] ?? "-")}");
            var table = new Table();
            table.AddColumn("Image");
            table.AddColumn("Vector (first values)");
            for (var i = 0; i < paths.Length; i++)
            {
                var head = string.Join(", ", result.Vectors[i].Take(4).Select(v => v.ToString("0.0000", CultureInfo.InvariantCulture)));
                table.AddRow(Markup.Escape(Path.GetFileName(paths[i])), $"{head}{(result.Vectors[i].Length > 4 ? ", …" : "")}");
            }
            AnsiConsole.Write(table);
            AnsiConsole.MarkupLine("[dim]Vectors are L2-normalised. --json gives them in full.[/]");
            return 0;
        });
        return command;
    }
}
