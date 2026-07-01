using System;
using System.Collections.Generic;
using MLoop.CLI.Commands;
using Xunit;

namespace MLoop.Tests.Commands;

/// <summary>
/// D9: dev-build API assembly resolution must prefer the most recently built DLL so a fresh
/// Release build wins over a stale Debug one. The old "first existing, Debug before Release"
/// order loaded the stale Debug assembly and hid Release-only fixes (cycle-104 had to set
/// MLOOP_API_PATH=Release to work around it).
/// </summary>
public class ServeCommandTests
{
    [Fact]
    public void NewestExisting_PrefersMostRecentlyBuilt()
    {
        var times = new Dictionary<string, DateTime>
        {
            ["Debug/MLoop.API.dll"] = new DateTime(2026, 1, 1),
            ["Release/MLoop.API.dll"] = new DateTime(2026, 6, 30), // newer build
        };

        var result = ServeCommand.NewestExisting(
            new[] { "Debug/MLoop.API.dll", "Release/MLoop.API.dll" },
            exists: _ => true,
            lastWriteUtc: p => times[p]);

        Assert.Equal("Release/MLoop.API.dll", result);
    }

    [Fact]
    public void NewestExisting_DebugNewer_PicksDebug()
    {
        // Symmetry: the rule is "newest", not "always Release" — a fresher Debug build wins too.
        var times = new Dictionary<string, DateTime>
        {
            ["Debug/MLoop.API.dll"] = new DateTime(2026, 6, 30),
            ["Release/MLoop.API.dll"] = new DateTime(2026, 1, 1),
        };

        var result = ServeCommand.NewestExisting(
            new[] { "Debug/MLoop.API.dll", "Release/MLoop.API.dll" },
            exists: _ => true,
            lastWriteUtc: p => times[p]);

        Assert.Equal("Debug/MLoop.API.dll", result);
    }

    [Fact]
    public void NewestExisting_SkipsMissingCandidates()
    {
        var result = ServeCommand.NewestExisting(
            new[] { "ghost/MLoop.API.dll", "Release/MLoop.API.dll" },
            exists: p => p.StartsWith("Release"),
            lastWriteUtc: _ => new DateTime(2026, 6, 30));

        Assert.Equal("Release/MLoop.API.dll", result);
    }

    [Fact]
    public void NewestExisting_NoneExist_ReturnsNull()
    {
        var result = ServeCommand.NewestExisting(
            new[] { "Debug/MLoop.API.dll", "Release/MLoop.API.dll" },
            exists: _ => false,
            lastWriteUtc: _ => DateTime.UtcNow);

        Assert.Null(result);
    }
}
