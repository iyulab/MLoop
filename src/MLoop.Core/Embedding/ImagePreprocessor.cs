using Microsoft.ML.Data;

namespace MLoop.Core.Embedding;

/// <summary>An image that could not be decoded — its position in the request, and why.</summary>
public sealed class UnreadableImageException(int index, string reason)
    : Exception($"Image {index} could not be read: {reason}")
{
    /// <summary>The image's zero-based position in the batch.</summary>
    public int Index { get; } = index;
}

/// <summary>
/// Turns an encoded image (PNG, JPEG, …) into an encoder's input — the steps of <see cref="ImagePreprocessing"/>, with
/// the arithmetic of the image processors encoders are published with.
/// </summary>
/// <remarks>
/// Those processors resize with Pillow, which resamples 8-bit images in fixed point (22 fractional bits) and rounds to
/// 8 bits between its horizontal and vertical passes. This does the same, so a vector computed here is the one the
/// encoder's own processor would give — a float resampler differs by a level here and there, and the difference reaches
/// the vector. Decoding is ML.NET's (<see cref="MLImage"/>).
/// </remarks>
public static class ImagePreprocessor
{
    private const int PrecisionBits = 32 - 8 - 2;
    private const double BicubicSupport = 2.0;

    /// <summary>The encoder input for one image: <c>3 × crop × crop</c> floats, channel-major (RGB).</summary>
    /// <exception cref="UnreadableImageException">The bytes are not an image ML.NET can decode.</exception>
    public static float[] ToTensor(ReadOnlySpan<byte> encoded, ImagePreprocessing preprocessing, int index = 0)
    {
        var (rgb, width, height) = Decode(encoded, index);
        var (newWidth, newHeight) = ResizedSize(width, height, preprocessing.ShortestEdge);
        var resized = Resize(rgb, width, height, newWidth, newHeight);
        return Normalize(resized, newWidth, newHeight, preprocessing);
    }

    /// <summary>The size an image is resized to: shortest edge <paramref name="shortestEdge"/>, the other in proportion, truncated.</summary>
    public static (int Width, int Height) ResizedSize(int width, int height, int shortestEdge)
    {
        var (shortSide, longSide) = width <= height ? (width, height) : (height, width);
        var newLong = (int)((double)shortestEdge * longSide / shortSide);
        return width <= height ? (shortestEdge, newLong) : (newLong, shortestEdge);
    }

    private static (byte[] Rgb, int Width, int Height) Decode(ReadOnlySpan<byte> encoded, int index)
    {
        MLImage image;
        try
        {
            using var stream = new MemoryStream(encoded.ToArray(), writable: false);
            image = MLImage.CreateFromStream(stream);
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            throw new UnreadableImageException(index, ex.GetBaseException().Message);
        }

        using (image)
        {
            var (r, b) = image.PixelFormat switch
            {
                MLPixelFormat.Rgba32 => (0, 2),
                MLPixelFormat.Bgra32 => (2, 0),
                var other => throw new UnreadableImageException(index, $"pixel format {other} is not supported"),
            };
            var pixels = image.Pixels;
            var rgb = new byte[image.Width * image.Height * 3];
            // Alpha is dropped, not composited — as converting to RGB does in the processors encoders ship with.
            for (int p = 0, o = 0; o < rgb.Length; p += 4, o += 3)
            {
                rgb[o] = pixels[p + r];
                rgb[o + 1] = pixels[p + 1];
                rgb[o + 2] = pixels[p + b];
            }
            return (rgb, image.Width, image.Height);
        }
    }

    private static byte[] Resize(byte[] rgb, int width, int height, int newWidth, int newHeight)
    {
        var horizontal = newWidth == width ? rgb : ResampleAxis(rgb, width, height, newWidth, alongRows: true);
        return newHeight == height ? horizontal : ResampleAxis(horizontal, newWidth, height, newHeight, alongRows: false);
    }

    /// <summary>One pass of Pillow's resampling: along each row (<paramref name="alongRows"/>) or each column.</summary>
    private static byte[] ResampleAxis(byte[] src, int width, int height, int outSize, bool alongRows)
    {
        var inSize = alongRows ? width : height;
        var (bounds, weights) = Coefficients(inSize, outSize);
        var outWidth = alongRows ? outSize : width;
        var outHeight = alongRows ? height : outSize;
        var dst = new byte[outWidth * outHeight * 3];
        var half = 1 << (PrecisionBits - 1);

        for (var y = 0; y < outHeight; y++)
        {
            for (var x = 0; x < outWidth; x++)
            {
                var o = alongRows ? x : y;
                var (first, count) = bounds[o];
                for (var c = 0; c < 3; c++)
                {
                    long sum = half;
                    for (var k = 0; k < count; k++)
                    {
                        var i = first + k;
                        var source = alongRows ? (y * width + i) * 3 + c : (i * width + x) * 3 + c;
                        sum += src[source] * (long)weights[o][k];
                    }
                    dst[(y * outWidth + x) * 3 + c] = (byte)Math.Clamp(sum >> PrecisionBits, 0, 255);
                }
            }
        }
        return dst;
    }

    /// <summary>Pillow's <c>precompute_coeffs</c> and <c>normalize_coeffs_8bpc</c> for its bicubic filter.</summary>
    private static ((int First, int Count)[] Bounds, int[][] Weights) Coefficients(int inSize, int outSize)
    {
        var scale = (double)inSize / outSize;
        var filterScale = Math.Max(scale, 1.0);
        var support = BicubicSupport * filterScale;
        var bounds = new (int, int)[outSize];
        var weights = new int[outSize][];

        for (var xx = 0; xx < outSize; xx++)
        {
            var center = (xx + 0.5) * scale;
            var min = Math.Max((int)(center - support + 0.5), 0);
            var max = Math.Min((int)(center + support + 0.5), inSize) - min;
            var raw = new double[max];
            var total = 0.0;
            for (var x = 0; x < max; x++)
            {
                raw[x] = Bicubic((x + min - center + 0.5) / filterScale);
                total += raw[x];
            }
            var fixedPoint = new int[max];
            for (var x = 0; x < max; x++)
            {
                var w = total != 0 ? raw[x] / total : raw[x];
                fixedPoint[x] = (int)(w * (1 << PrecisionBits) + (w < 0 ? -0.5 : 0.5));
            }
            bounds[xx] = (min, max);
            weights[xx] = fixedPoint;
        }
        return (bounds, weights);
    }

    private static double Bicubic(double x)
    {
        const double a = -0.5;
        x = Math.Abs(x);
        if (x < 1.0)
            return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
        if (x < 2.0)
            return (((x - 5) * x + 8) * x - 4) * a;
        return 0.0;
    }

    private static float[] Normalize(byte[] rgb, int width, int height, ImagePreprocessing p)
    {
        var crop = p.CropSize;
        var top = (height - crop) / 2;
        var left = (width - crop) / 2;
        var tensor = new float[3 * crop * crop];
        const float rescale = 1f / 255f;
        for (var c = 0; c < 3; c++)
        {
            var mean = p.Mean[c];
            var std = p.Std[c];
            for (var y = 0; y < crop; y++)
            {
                for (var x = 0; x < crop; x++)
                {
                    var value = rgb[((top + y) * width + left + x) * 3 + c] * rescale;
                    tensor[(c * crop + y) * crop + x] = (value - mean) / std;
                }
            }
        }
        return tensor;
    }
}
