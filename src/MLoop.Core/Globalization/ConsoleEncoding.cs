using System.Text;

namespace MLoop.Core.Globalization;

/// <summary>
/// The single authority for the encoding MLoop writes its console output in.
/// </summary>
/// <remarks>
/// <para>
/// A Windows console writes in the machine's OEM code page unless told otherwise — 949 on Korean Windows — and a character
/// outside it is not an error but a <c>?</c>. MLoop's messages carry a few (<c>—</c> in most of its sentences, <c>✓</c> in the
/// Production column of <c>mloop status</c>), so on the machine this tool is aimed at a user read «? exp-001» where the
/// model is in production, and a program that reads our output as UTF-8 (a Node process, a PowerShell with UTF-8 output) got
/// Korean column names as bytes it could not decode. Everywhere else the console is UTF-8 already.
/// </para>
/// <para>
/// So on Windows the process writes UTF-8 for as long as it runs and puts the console back as it found it: the code page
/// belongs to the console window, not to this process, and leaving it changed would garble the next program that
/// writes the old code page there (the way <c>dotnet</c> itself does it). <c>--json</c> is unaffected — it
/// escapes every non-ASCII character, so a machine reader gets the same bytes in any code page.
/// </para>
/// </remarks>
public static class ConsoleEncoding
{
    private const int Utf8CodePage = 65001;

    /// <summary>
    /// Make the console write UTF-8 until the returned value is disposed; <c>null</c> when nothing was changed (not Windows, already
    /// UTF-8, or no console to read an encoding from).
    /// </summary>
    public static IDisposable? Utf8OnWindows()
        => Apply(OperatingSystem.IsWindows(), () => Console.OutputEncoding, encoding => Console.OutputEncoding = encoding);

    /// <summary>The same, with the platform and the console passed in — what the tests drive.</summary>
    public static IDisposable? Apply(bool isWindows, Func<Encoding> get, Action<Encoding> set)
    {
        ArgumentNullException.ThrowIfNull(get);
        ArgumentNullException.ThrowIfNull(set);
        if (!isWindows)
            return null;

        Encoding current;
        try
        {
            current = get();
        }
        catch (IOException)
        {
            // No console behind this process (a service, a detached run): there is no code page to change.
            return null;
        }

        if (current.CodePage == Utf8CodePage)
            return null;

        try
        {
            // Without a byte order mark: a console that echoes it shows it.
            set(new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
        }
        catch (IOException)
        {
            return null;
        }
        return new Restore(current, set);
    }

    private sealed class Restore(Encoding previous, Action<Encoding> set) : IDisposable
    {
        public void Dispose()
        {
            try
            {
                set(previous);
            }
            catch (IOException)
            {
                // The console went away first; nothing is left to put back.
            }
        }
    }
}
