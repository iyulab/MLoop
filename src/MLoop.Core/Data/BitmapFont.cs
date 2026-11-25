using SkiaSharp;

namespace MLoop.Core.Data;

/// <summary>
/// The printable ASCII characters as 5 × 7 dots, for the labels <see cref="PictureOverlay"/> writes on pictures. Each character
/// is five columns, left to right; in a column the lowest bit is the top dot.
/// </summary>
internal static class BitmapFont
{
    private const string Columns =
        "0000000000" + "00005F0000" + "0007000700" + "147F147F14" + "242A7F2A12" + "2313086462" + "3649552250" + "0005030000" + // space ! " # $ % & '
        "001C224100" + "0041221C00" + "082A1C2A08" + "08083E0808" + "0050300000" + "0808080808" + "0060600000" + "2010080402" + // ( ) * + , - . /
        "3E5149453E" + "00427F4000" + "4261514946" + "2141454B31" + "1814127F10" + "2745454539" + "3C4A494930" + "0171090503" + // 0-7
        "3649494936" + "064949291E" + "0036360000" + "0056360000" + "0008142241" + "1414141414" + "4122140800" + "0201510906" + // 8 9 : ; < = > ?
        "324979413E" + "7E1111117E" + "7F49494936" + "3E41414122" + "7F4141221C" + "7F49494941" + "7F09090101" + "3E41415132" + // @ A-G
        "7F0808087F" + "00417F4100" + "2040413F01" + "7F08142241" + "7F40404040" + "7F0204027F" + "7F0408107F" + "3E4141413E" + // H-O
        "7F09090906" + "3E4151215E" + "7F09192946" + "4649494931" + "01017F0101" + "3F4040403F" + "1F2040201F" + "7F2018207F" + // P-W
        "6314081463" + "0304780403" + "6151494543" + "00007F4141" + "0204081020" + "41417F0000" + "0402010204" + "4040404040" + // X Y Z [ \ ] ^ _
        "0001020400" + "2054545478" + "7F48444438" + "3844444420" + "384444487F" + "3854545418" + "087E090102" + "081454543C" + // ` a-g
        "7F08040478" + "00447D4000" + "2040443D00" + "007F102844" + "00417F4000" + "7C04180478" + "7C08040478" + "3844444438" + // h-o
        "7C14141408" + "081414187C" + "7C08040408" + "4854545420" + "043F444020" + "3C4040207C" + "1C2040201C" + "3C4030403C" + // p-w
        "4428102844" + "0C5050503C" + "4464544C44" + "0008364100" + "00007F0000" + "0041360800" + "0804081008";                  // x y z { | } ~

    private static readonly byte[] Dots = Convert.FromHexString(Columns);

    public static bool Has(char c) => c is >= ' ' and <= '~';

    /// <summary>The width of <paramref name="text"/> at <paramref name="unit"/> pixels a dot, with a dot of margin at each end.</summary>
    public static float Width(string text, int unit) => (text.Length * 6 + 1) * unit;

    /// <summary>The height of a line, with a dot of margin above and below.</summary>
    public static float Height(int unit) => 9 * unit;

    public static void Draw(SKCanvas canvas, string text, float x, float y, int unit)
    {
        using var ink = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill, IsAntialias = false };
        for (var i = 0; i < text.Length; i++)
        {
            var glyph = (text[i] - ' ') * 5;
            for (var column = 0; column < 5; column++)
                for (var row = 0; row < 7; row++)
                    if ((Dots[glyph + column] & (1 << row)) != 0)
                        canvas.DrawRect(SKRect.Create(x + (1 + i * 6 + column) * unit, y + (1 + row) * unit, unit, unit), ink);
        }
    }
}
