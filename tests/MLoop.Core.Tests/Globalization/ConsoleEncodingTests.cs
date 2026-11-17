using System.Text;
using MLoop.Core.Globalization;

namespace MLoop.Core.Tests.Globalization;

/// <summary>
/// On Windows the console writes its OEM code page (949 on Korean Windows) and a character outside it prints as «?»; the
/// process writes UTF-8 for its run and puts the console back after. These drive the rule with a fake console — the
/// real one is process-global state, and only Windows has the problem.
/// </summary>
public class ConsoleEncodingTests
{
    private sealed class FakeConsole(Encoding initial)
    {
        public Encoding Current { get; private set; } = initial;
        public List<int> Sets { get; } = [];
        public Encoding Get() => Current;
        public void Set(Encoding encoding) { Current = encoding; Sets.Add(encoding.CodePage); }
    }

    private static Encoding Korean()
    {
        Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
        return Encoding.GetEncoding(949);
    }

    [Fact]
    public void OnWindowsWithAKoreanConsole_UTF8IsWritten_AndTheConsoleIsPutBackAfter()
    {
        var console = new FakeConsole(Korean());

        using (var scope = ConsoleEncoding.Apply(isWindows: true, console.Get, console.Set))
        {
            Assert.NotNull(scope);
            Assert.Equal(65001, console.Current.CodePage);
            Assert.Empty(console.Current.GetPreamble());
            // What the change is for: the dash and the check mark survive, where the code page turned them into «?».
            Assert.Equal("—✓", Encoding.UTF8.GetString(console.Current.GetBytes("—✓")));
        }

        Assert.Equal(949, console.Current.CodePage);
        Assert.Equal([65001, 949], console.Sets);
    }

    [Fact]
    public void TheCodePageItReplaces_CouldNotWriteTheCharactersTheMessagesUse()
    {
        // The reason for the whole thing, as a measurement: both are «?» in 949.
        var korean = Korean();
        Assert.Equal("?", korean.GetString(korean.GetBytes("—")));
        Assert.Equal("?", korean.GetString(korean.GetBytes("✓")));
        Assert.Equal("온도", korean.GetString(korean.GetBytes("온도")));
    }

    [Fact]
    public void AConsoleThatWritesUTF8Already_IsLeftAlone()
    {
        var console = new FakeConsole(new UTF8Encoding(false));

        var scope = ConsoleEncoding.Apply(isWindows: true, console.Get, console.Set);

        Assert.Null(scope);
        Assert.Empty(console.Sets);
    }

    [Fact]
    public void OffWindows_NothingIsTouched()
    {
        var console = new FakeConsole(Korean());

        var scope = ConsoleEncoding.Apply(isWindows: false, console.Get, console.Set);

        Assert.Null(scope);
        Assert.Empty(console.Sets);
    }

    [Fact]
    public void WithNoConsoleBehindTheProcess_NothingIsChanged_AndNothingThrows()
    {
        var sets = new List<Encoding>();

        var scope = ConsoleEncoding.Apply(isWindows: true, () => throw new IOException("The handle is invalid."), sets.Add);

        Assert.Null(scope);
        Assert.Empty(sets);
    }

    [Fact]
    public void AConsoleThatGoesAwayBeforeTheRestore_DoesNotFailTheRun()
    {
        var console = new FakeConsole(Korean());
        var calls = 0;
        var scope = ConsoleEncoding.Apply(true, console.Get, e => { if (++calls > 1) throw new IOException("gone"); console.Set(e); });

        var ex = Record.Exception(() => scope!.Dispose());

        Assert.Null(ex);
    }
}
