using System.Text.RegularExpressions;
using MLoop.CLI.Commands;

namespace MLoop.Tests.Commands;

/// <summary>
/// The generated container artifacts are the one product output no unit test can execute, so five
/// string assertions about them passed for as long as the Dockerfile existed — including the whole
/// time it could not build at all (the final stage added the Microsoft *Debian 12* apt
/// feed to what is an *Ubuntu* base image and installed the full SDK on top of a runtime that
/// already had one, and `apt-get` exited 100). Asserting that a base-image name appears cannot see
/// that. These checks encode the three failures that were actually measured, and each one is paired
/// with a self-check that feeds it the exact shape it forbids — a matcher that has stopped matching
/// reports a clean tree just as loudly as a clean tree does.
/// </summary>
public class GeneratedContainerAuditTests
{
    /// <summary>The Dockerfile as it stood when `docker build` failed — the self-checks' input.</summary>
    private const string HistoricalBrokenDockerfile = """
        FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS base
        FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
        FROM base AS final
        RUN apt-get update && \
            apt-get install -y wget && \
            wget https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb -O packages-microsoft-prod.deb && \
            dpkg -i packages-microsoft-prod.deb && \
            apt-get install -y dotnet-sdk-10.0
        HEALTHCHECK CMD wget --quiet --tries=1 --spider http://localhost:5000/health || exit 1
        """;

    private static IReadOnlyList<string> Audit(string dockerfile)
    {
        var findings = new List<string>();

        // A hardcoded distro feed is wrong the moment the base image's distro moves, and nothing in
        // the file records which distro it assumed. The runtime image is Ubuntu; this said Debian.
        if (Regex.IsMatch(dockerfile, @"packages\.microsoft\.com/config/"))
            findings.Add("adds a distro-pinned Microsoft package feed");

        // The runtime image already carries Microsoft.NETCore.App and Microsoft.AspNetCore.App.
        // Installing a .NET distribution over it is redundant at best and unresolvable at worst.
        if (Regex.IsMatch(dockerfile, @"(dotnet-sdk|dotnet-runtime)-\d"))
            findings.Add("installs a .NET distribution through the package manager");

        // `--spider` is a HEAD request. It is only a valid probe if the endpoint answers HEAD.
        if (dockerfile.Contains("--spider"))
            findings.Add("probes with HEAD via --spider");

        return findings;
    }

    [Fact]
    public void GeneratedDockerfile_PassesTheAudit()
    {
        Assert.Empty(Audit(DockerCommand.GenerateDockerfile("default", 5000)));
    }

    [Fact]
    public void Audit_SeesEveryFailureItExistsFor()
    {
        // Self-check: without this, a matcher that quietly stopped matching would report the
        // generated Dockerfile clean and sound exactly like a real pass.
        Assert.Equal(3, Audit(HistoricalBrokenDockerfile).Count);
    }

    [Fact]
    public void Audit_DoesNotRefuseALegitimateAptLayer()
    {
        // The remaining apt layer installs wget for the HEALTHCHECK. The audit must not read an
        // ordinary package install as one of the failures above, or it would block the fix.
        Assert.Empty(Audit("RUN apt-get update && apt-get install -y --no-install-recommends wget"));
    }

    [Fact]
    public void GeneratedCompose_RequiresASigningKeyWhenItAsksForProduction()
    {
        // Production refuses to start on the default JWT key. The generated compose file set the
        // environment and not the key, so `docker compose up` — the command the generator's own
        // usage text offers — crash-looped the container (7 restarts, measured against a running container). Compose
        // has to *demand* the value, not default it: a placeholder secret in a generated file is
        // worse than a missing one.
        var compose = DockerCommand.GenerateDockerCompose("default", 5000);

        if (!compose.Contains("ASPNETCORE_ENVIRONMENT=Production"))
            return; // Not claiming Production; the key is then not required.

        Assert.Matches(@"Jwt__Key=\$\{[A-Z_]+:\?", compose);
    }

    [Fact]
    public void GeneratedCompose_DoesNotCarryTheObsoleteVersionKey()
    {
        // Compose v2 warns on `version:`; a generated file should not open with a warning.
        Assert.DoesNotMatch(@"(?m)^version:", DockerCommand.GenerateDockerCompose("default", 5000));
    }
}
