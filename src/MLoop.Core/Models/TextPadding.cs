using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLoop.Core.Models;

/// <summary>
/// One more word at the end of a text, in front of a pretrained text model that cannot say anything about the last
/// word of what it is given (<see cref="NerAlignment"/>, <see cref="AnswerAlignment"/>): both of ML.NET's token-level
/// text models drop or misplace the answer at the last token, and the successor word is what that token is read against.
/// </summary>
public static class TextPadding
{
    /// <summary>The word that follows the last real word.</summary>
    public const string PadWord = ".";

    /// <summary><paramref name="text"/> with the word that gives its last word a successor.</summary>
    public static string Padded(string? text) => $"{text} {PadWord}";

    /// <summary>A text <see cref="Padded"/> by this step, as it was.</summary>
    public static string Unpadded(string? text)
    {
        var suffix = $" {PadWord}";
        return text is not null && text.EndsWith(suffix, StringComparison.Ordinal) ? text[..^suffix.Length] : text ?? "";
    }

    /// <summary>A step that replaces the text column <paramref name="column"/> by <paramref name="map"/> of it, row by row.</summary>
    internal static ITransformer Mapping(MLContext mlContext, string column, Func<string?, string> map)
    {
        var input = SchemaDefinition.Create(typeof(TextCell));
        input[0].ColumnName = column;
        var output = SchemaDefinition.Create(typeof(TextCell));
        output[0].ColumnName = column;
        return mlContext.Transforms.CustomMapping<TextCell, TextCell>(
                (row, result) => result.Text = map(row.Text), contractName: null, input, output)
            .Fit(mlContext.Data.LoadFromEnumerable(Array.Empty<TextCell>(), input));
    }

    private sealed class TextCell
    {
        public string? Text { get; set; }
    }
}
