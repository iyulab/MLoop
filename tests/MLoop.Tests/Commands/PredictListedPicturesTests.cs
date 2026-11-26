using MLoop.CLI.Commands;

namespace MLoop.Tests.Commands;

/// <summary>
/// The pictures a classifier's table names are read before the model is: its loader reads a file it cannot open as nothing
/// and still answers. A folder becomes such a table, and a user may write one.
/// </summary>
public sealed class PredictListedPicturesTests : IDisposable
{
    private readonly string _root = Directory.CreateTempSubdirectory("mloop-listed-pictures-").FullName;

    public void Dispose() => Directory.Delete(_root, recursive: true);

    private string Table(params string[] lines)
    {
        var path = Path.Combine(_root, "list.csv");
        File.WriteAllLines(path, lines, new System.Text.UTF8Encoding(encoderShouldEmitUTF8Identifier: true));
        return path;
    }

    private string Picture(string name)
    {
        var path = Path.Combine(_root, name);
        using var bitmap = new SkiaSharp.SKBitmap(4, 4);
        using var data = bitmap.Encode(SkiaSharp.SKEncodedImageFormat.Png, 100);
        File.WriteAllBytes(path, data.ToArray());
        return path;
    }

    [Fact]
    public async Task A_listed_file_that_is_not_a_picture_or_not_there_is_named()
    {
        var good = Picture("good, one.png"); // a comma, quoted in the table
        var broken = Path.Combine(_root, "broken.png");
        File.WriteAllText(broken, "text");
        var absent = Path.Combine(_root, "absent.png");

        var refusal = await PredictCommand.UnreadableListedPicturesAsync(Table("ImagePath", $"\"{good}\"", broken, absent));

        Assert.NotNull(refusal);
        Assert.Contains(broken, refusal, StringComparison.Ordinal);
        Assert.Contains(absent, refusal, StringComparison.Ordinal);
        Assert.DoesNotContain(good, refusal, StringComparison.Ordinal);
    }

    [Fact]
    public async Task A_table_of_pictures_that_open_passes_and_one_with_no_picture_column_is_left_to_the_schema_check()
    {
        var good = Picture("good.png");

        Assert.Null(await PredictCommand.UnreadableListedPicturesAsync(Table("ImagePath", good)));
        Assert.Null(await PredictCommand.UnreadableListedPicturesAsync(Table("Path", "nowhere.png")));
    }
}
