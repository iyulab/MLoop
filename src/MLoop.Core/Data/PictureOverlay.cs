using System.Globalization;
using MLoop.Core.Embedding;
using SkiaSharp;

namespace MLoop.Core.Data;

/// <summary>An object a detector found, to be drawn: its label, score and box in the picture's pixels.</summary>
public sealed record DrawnBox(string Label, float Score, float X0, float Y0, float X1, float Y1);

/// <summary>
/// Copies of pictures with what a model found drawn onto them — a detector's boxes with their labels and scores, or how far each
/// patch of a picture is from what an anomaly model learned as normal. The originals are only read.
/// </summary>
public static class PictureOverlay
{
    private static readonly SKColor Ink = new(17, 17, 17);
    private static readonly SKColor Heat = new(224, 69, 42);

    /// <summary>
    /// Where the copy of each of <paramref name="sources"/> goes under <paramref name="targetFolder"/>: its path under
    /// <paramref name="sourceRoot"/> with a <c>.png</c> extension, or with <c>.png</c> after its own extension when two pictures
    /// would otherwise share a name (<c>a.jpg</c> beside <c>a.png</c>).
    /// </summary>
    public static IReadOnlyList<string> Targets(IReadOnlyList<string> sources, string sourceRoot, string targetFolder)
    {
        ArgumentNullException.ThrowIfNull(sources);
        var relative = sources.Select(s => Path.GetRelativePath(sourceRoot, s)).ToList();
        var shared = relative
            .GroupBy(r => Path.ChangeExtension(r, ".png"), StringComparer.OrdinalIgnoreCase)
            .Where(g => g.Count() > 1)
            .Select(g => g.Key)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
        return [.. relative.Select(r =>
        {
            var name = Path.ChangeExtension(r, ".png");
            return Path.GetFullPath(Path.Combine(targetFolder, shared.Contains(name) ? r + ".png" : name));
        })];
    }

    /// <summary>
    /// Why the copies cannot go into <paramref name="targetFolder"/>, or <c>null</c> when they can: it is a file; a copy would
    /// write over one of the pictures being read; or — when a whole folder is being read (<paramref name="readFolder"/>) — it is
    /// inside that folder, so that the next run would take the copies for more pictures.
    /// </summary>
    public static string? TargetRefusal(IReadOnlyList<string> sources, string? readFolder, string sourceRoot, string targetFolder)
    {
        ArgumentNullException.ThrowIfNull(sources);
        var comparison = OperatingSystem.IsLinux() ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase;
        var target = Path.TrimEndingDirectorySeparator(Path.GetFullPath(targetFolder));
        if (File.Exists(target))
            return $"'{targetFolder}' is a file; name a folder for the pictures.";

        if (readFolder is not null)
        {
            var read = Path.TrimEndingDirectorySeparator(Path.GetFullPath(readFolder));
            if (target.Equals(read, comparison) || target.StartsWith(read + Path.DirectorySeparatorChar, comparison))
                return $"'{targetFolder}' is inside the folder of pictures being read ('{readFolder}'): the copies would be " +
                       "read as pictures the next time. Name a folder outside it.";
        }

        var originals = sources.Select(Path.GetFullPath).ToHashSet(StringComparer.FromComparison(comparison));
        return Targets(sources, sourceRoot, targetFolder).FirstOrDefault(originals.Contains) is { } overwritten
            ? $"'{targetFolder}' would write a copy over the picture '{overwritten}'. Name another folder."
            : null;
    }

    /// <summary>Writes a copy of <paramref name="source"/> with each of <paramref name="boxes"/> outlined and named.</summary>
    public static void SaveBoxes(string source, IReadOnlyList<DrawnBox> boxes, string target)
    {
        ArgumentNullException.ThrowIfNull(boxes);
        using var bitmap = Load(source);
        using (var canvas = new SKCanvas(bitmap))
        {
            var unit = Unit(bitmap);
            foreach (var box in boxes)
            {
                var rect = new SKRect(box.X0, box.Y0, box.X1, box.Y1);
                Outline(canvas, rect, unit);
                var caption = string.Create(CultureInfo.InvariantCulture, $"{box.Label} {box.Score:0.00}");
                Tag(canvas, bitmap, caption, rect.Left, rect.Top, unit);
            }
        }

        Save(bitmap, target);
    }

    /// <summary>
    /// Writes a copy of <paramref name="source"/> with a heat map over the part of it the encoder read (<paramref name="crop"/>):
    /// each patch of the <paramref name="rows"/> × <paramref name="columns"/> grid is tinted by how far its distance is beyond
    /// <paramref name="normalEnd"/> — where the held-out normal pictures ended — so a normal picture stays clear. The strongest
    /// patch is outlined when it is beyond that end, and the picture's score is written on it.
    /// </summary>
    public static void SaveHeatMap(string source, IReadOnlyList<float> distances, int rows, int columns, ImageRegion crop,
        double? normalEnd, string target)
    {
        ArgumentNullException.ThrowIfNull(distances);
        ArgumentNullException.ThrowIfNull(crop);
        if (distances.Count != rows * columns || distances.Count == 0)
            throw new ArgumentException($"{distances.Count} distances do not fill a {rows} x {columns} grid.", nameof(distances));

        using var bitmap = Load(source);
        using (var canvas = new SKCanvas(bitmap))
        {
            var strongest = 0;
            for (var i = 1; i < distances.Count; i++)
                if (distances[i] > distances[strongest])
                    strongest = i;
            double score = distances[strongest];
            // Clear up to where normal ended; full colour at twice that, or at the picture's own strongest patch when further.
            var floor = normalEnd ?? distances.Min();
            var top = Math.Max(score, floor * 2);
            var patchWidth = (float)((crop.X1 - crop.X0) / columns);
            var patchHeight = (float)((crop.Y1 - crop.Y0) / rows);
            SKRect Patch(int i) => SKRect.Create(
                (float)crop.X0 + i % columns * patchWidth, (float)crop.Y0 + i / columns * patchHeight, patchWidth, patchHeight);

            for (var i = 0; i < distances.Count; i++)
            {
                var strength = top > floor ? Math.Clamp((distances[i] - floor) / (top - floor), 0, 1) : 0;
                if (strength <= 0)
                    continue;
                using var tint = new SKPaint { Color = Heat.WithAlpha((byte)(40 + 150 * strength)), Style = SKPaintStyle.Fill };
                canvas.DrawRect(Patch(i), tint);
            }

            var unit = Unit(bitmap);
            // A picture within what normal showed has no place to point at; with no normal end recorded, the strongest patch is shown.
            if (normalEnd is not { } limit || score > limit)
                Outline(canvas, Patch(strongest), unit);
            var caption = normalEnd is { } end
                ? string.Create(CultureInfo.InvariantCulture, $"{score:0.000} (normal to {end:0.000})")
                : string.Create(CultureInfo.InvariantCulture, $"{score:0.000}");
            Tag(canvas, bitmap, caption, 0, 0, unit);
        }

        Save(bitmap, target);
    }

    // One pixel of line and text per 320 pixels of the picture's shorter side, so a small picture is not covered and a large one
    // is still read.
    private static int Unit(SKBitmap bitmap) => Math.Max(1, (int)Math.Round(Math.Min(bitmap.Width, bitmap.Height) / 320.0));

    private static SKBitmap Load(string source)
    {
        using var decoded = SKBitmap.Decode(source)
            ?? throw new InvalidOperationException($"'{source}' could not be read as a picture.");
        // Drawn on an opaque RGBA copy: an indexed or grey picture takes colour, and the original is not touched.
        var copy = new SKBitmap(new SKImageInfo(decoded.Width, decoded.Height, SKColorType.Rgba8888, SKAlphaType.Premul));
        using var canvas = new SKCanvas(copy);
        canvas.Clear(SKColors.White);
        canvas.DrawBitmap(decoded, 0, 0);
        return copy;
    }

    private static void Save(SKBitmap bitmap, string target)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(target))!);
        using var data = bitmap.Encode(SKEncodedImageFormat.Png, 100);
        using var stream = File.Create(target);
        data.SaveTo(stream);
    }

    // A dark line with a light one around it, so that it shows on any picture.
    private static void Outline(SKCanvas canvas, SKRect rect, int unit)
    {
        using var light = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Stroke, StrokeWidth = unit * 4 };
        using var dark = new SKPaint { Color = Ink, Style = SKPaintStyle.Stroke, StrokeWidth = unit * 2 };
        canvas.DrawRect(rect, light);
        canvas.DrawRect(rect, dark);
    }

    /// <summary>Light text on a dark tag above the point (<paramref name="x"/>, <paramref name="y"/>), or below it at the top edge.</summary>
    private static void Tag(SKCanvas canvas, SKBitmap bitmap, string text, float x, float y, int unit)
    {
        var (width, height, draw) = Lettering(text, unit);
        var left = Math.Clamp(x - unit, 0, Math.Max(0, bitmap.Width - width));
        var top = y - height - unit >= 0 ? y - height - unit : Math.Min(y + unit, Math.Max(0, bitmap.Height - height));
        using var back = new SKPaint { Color = Ink, Style = SKPaintStyle.Fill };
        canvas.DrawRect(SKRect.Create(left, top, width, height), back);
        draw(canvas, left, top);
    }

    // Letters are drawn from a built-in 5 x 7 font: the imaging library finds no fonts at all on a bare Linux install. A label
    // with characters outside it (Hangul, say) is drawn with a font of the machine's when one has them.
    private static (float Width, float Height, Action<SKCanvas, float, float> Draw) Lettering(string text, int unit)
    {
        if (text.All(BitmapFont.Has))
            return (BitmapFont.Width(text, unit), BitmapFont.Height(unit), (canvas, x, y) => BitmapFont.Draw(canvas, text, x, y, unit));

        var typeface = text.Where(c => !BitmapFont.Has(c)).Select(c => SKFontManager.Default.MatchCharacter(c)).FirstOrDefault(t => t is not null);
        if (typeface is not null)
        {
            var paint = new SKPaint { Typeface = typeface, TextSize = unit * 10, Color = SKColors.White, IsAntialias = true };
            if (paint.ContainsGlyphs(text))
                return (paint.MeasureText(text) + unit * 2, unit * 12, (canvas, x, y) => canvas.DrawText(text, x + unit, y + unit * 10, paint));
            paint.Dispose();
        }

        // No font here has them: what the built-in one can draw is kept, and each other character becomes '?'.
        var plain = new string([.. text.Select(c => BitmapFont.Has(c) ? c : '?')]);
        return (BitmapFont.Width(plain, unit), BitmapFont.Height(unit), (canvas, x, y) => BitmapFont.Draw(canvas, plain, x, y, unit));
    }
}
