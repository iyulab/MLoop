using System.CommandLine;
using System.Globalization;
using System.Text.Json;
using MLoop.CLI.Infrastructure.Diagnostics;
using MLoop.Core.Embedding;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// <c>mloop embed &lt;images…&gt;</c> / <c>mloop embed --text …</c> — one vector per image, or per text, from an installed
/// pretrained encoder, with the encoder's identity (id and weights' SHA-256) that a consumer keeps beside them. Needs no
/// project.
/// </summary>
public static class EmbedCommand
{
    // Absent fields are left out, as `POST /embed` leaves them out — one shape for both.
    private static readonly JsonSerializerOptions Json = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull,
    };

    public static Command Create()
    {
        var imagesArg = new Argument<string[]>("images")
        {
            Description = $"Image files (PNG, JPEG, …), at most {EmbedderPool.MaxImages}",
            Arity = ArgumentArity.ZeroOrMore,
        };
        var textOption = new Option<string[]>("--text")
        {
            Description = "A text to embed with a text encoder, instead of images; give it once per text",
            AllowMultipleArgumentsPerToken = false,
        };
        var textFileOption = new Option<string?>("--text-file")
        {
            Description = $"A UTF-8 file of texts to embed, one per line (at most {EmbedderPool.MaxTexts} with --text)",
        };
        var encoderOption = new Option<string?>("--encoder") { Description = "The encoder to use; may be left out when only one is installed" };
        var jsonOption = new Option<bool>("--json") { Description = "Emit the encoder and every vector as JSON to stdout" };
        var patchesOption = new Option<bool>("--patches")
        {
            Description = $"Also one vector per patch of each image, on the encoder's grid (at most {EmbedderPool.MaxPatchImages} images)",
        };

        var command = new Command("embed", "Embed images or texts as vectors with a pretrained encoder");
        command.Arguments.Add(imagesArg);
        command.Options.Add(textOption);
        command.Options.Add(textFileOption);
        command.Options.Add(encoderOption);
        command.Options.Add(jsonOption);
        command.Options.Add(patchesOption);

        command.SetAction(parseResult =>
        {
            var paths = parseResult.GetValue(imagesArg) ?? [];
            var json = parseResult.GetValue(jsonOption);
            using var machineOutput = json ? new JsonOutputScope() : null;
            var texts = parseResult.GetValue(textOption) ?? [];
            var textFile = parseResult.GetValue(textFileOption);
            if (texts.Length > 0 || textFile is not null)
            {
                if (paths.Length > 0 || parseResult.GetValue(patchesOption))
                {
                    ErrorConsole.Error("Give images or texts (--text, --text-file), not both; --patches is for images.");
                    return 1;
                }
                return EmbedTexts(parseResult.GetValue(encoderOption), texts, textFile, json);
            }
            if (paths.Length == 0)
            {
                ErrorConsole.Error("Nothing to embed: give image files, or texts with --text or --text-file.");
                return 1;
            }
            if (paths.FirstOrDefault(p => !File.Exists(p)) is { } missing)
            {
                ErrorConsole.Error(Markup.Escape($"Image not found: {missing}"));
                return 1;
            }

            using var pool = new EmbedderPool(new EncoderStore());
            EmbeddingResult? result;
            string? refusal;
            try
            {
                (result, refusal) = pool.Embed(parseResult.GetValue(encoderOption), [.. paths.Select(File.ReadAllBytes)], parseResult.GetValue(patchesOption));
            }
            catch (UnreadableImageException ex)
            {
                ErrorConsole.Error(Markup.Escape($"{paths[ex.Index]}: {ex.Message}"));
                return 1;
            }
            if (result is null)
            {
                ErrorConsole.Error(Markup.Escape(refusal!));
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
            if (result.Grid is { } grid)
                AnsiConsole.MarkupLine($"Patches: {grid.Rows} × {grid.Columns} of {grid.PatchSize} px over the {grid.CropSize} px centre crop, per image.");
            AnsiConsole.MarkupLine("[dim]Vectors are L2-normalised. --json gives them in full.[/]");
            return 0;
        });
        return command;
    }

    private static int EmbedTexts(string? encoder, string[] texts, string? textFile, bool json)
    {
        var all = texts.ToList();
        if (textFile is not null)
        {
            if (!File.Exists(textFile))
            {
                ErrorConsole.Error(Markup.Escape($"Text file not found: {textFile}"));
                return 1;
            }
            all.AddRange(File.ReadLines(textFile).Where(line => line.Length > 0));
        }

        using var pool = new EmbedderPool(new EncoderStore());
        var (result, refusal) = pool.EmbedTexts(encoder, all);
        if (result is null)
        {
            ErrorConsole.Error(Markup.Escape(refusal!));
            return 1;
        }
        if (result.Truncated is > 0 and var truncated)
            WarningConsole.Warn(Markup.Escape($"{truncated} of {all.Count} texts are longer than encoder '{result.Encoder.Id}' reads; their vectors are of their beginning."));

        if (json)
        {
            // POST /embed's answer, plus the truncation warning the document's reader would otherwise not see.
            Console.WriteLine(JsonSerializer.Serialize(
                new { result.Encoder, result.Vectors, result.Truncated, Warnings = JsonOutputScope.Warnings }, Json));
            return 0;
        }

        var e = result.Encoder;
        AnsiConsole.MarkupLine($"Encoder [cyan]{Markup.Escape(e.Id)}[/] {Markup.Escape(e.Version)} · {e.Dimension} dimensions · SHA-256 {Markup.Escape(e.Sha256?[..12] ?? "-")}");
        var table = new Table();
        table.AddColumn("Text");
        table.AddColumn("Vector (first values)");
        for (var i = 0; i < all.Count; i++)
        {
            var head = string.Join(", ", result.Vectors[i].Take(4).Select(v => v.ToString("0.0000", CultureInfo.InvariantCulture)));
            var shown = all[i].Length > 40 ? all[i][..40] + "…" : all[i];
            table.AddRow(Markup.Escape(shown), $"{head}{(result.Vectors[i].Length > 4 ? ", …" : "")}");
        }
        AnsiConsole.Write(table);
        AnsiConsole.MarkupLine("[dim]Vectors are L2-normalised. --json gives them in full.[/]");
        return 0;
    }
}
