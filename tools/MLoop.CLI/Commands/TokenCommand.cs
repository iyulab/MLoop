using System.CommandLine;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;
using Microsoft.IdentityModel.Tokens;
using MLoop.Core.Security;
using Spectre.Console;

namespace MLoop.CLI.Commands;

/// <summary>
/// Issues a JWT bearer token for calling a local <c>mloop serve</c> API. Every endpoint except
/// <c>/health</c> requires authentication, but the server has no token-issuance surface — without
/// this command there is no way to obtain a token, so a freshly served model is unreachable.
/// Signs with <see cref="DevJwtDefaults"/> (the same authority the API validates against) so issued
/// tokens validate out of the box; <c>--key</c> overrides for a server started with a custom Jwt:Key.
/// Not for production auth — it's a local-serving convenience.
/// </summary>
public class TokenCommand : Command
{
    public TokenCommand() : base("token", "Issue a JWT bearer token for the local serve API (Authorization: Bearer)")
    {
        var roleOption = new Option<string?>("--role", "-r")
        {
            Description = "Role claim — use 'admin' for write endpoints (promote/train/evaluate/cache). Omit for read-only/predict."
        };
        var subjectOption = new Option<string>("--subject", "-s")
        {
            Description = "Subject (sub) claim identifying the caller",
            DefaultValueFactory = _ => "mloop-cli"
        };
        var hoursOption = new Option<int>("--expires-hours")
        {
            Description = "Token lifetime in hours",
            DefaultValueFactory = _ => 24
        };
        var keyOption = new Option<string?>("--key")
        {
            Description = "Override signing key (must match the server's Jwt:Key, >=32 chars). Defaults to the dev key."
        };
        var quietOption = new Option<bool>("--quiet", "-q")
        {
            Description = "Print only the raw token — for scripting, e.g. export MLOOP_TOKEN=$(mloop token -q)"
        };

        Options.Add(roleOption);
        Options.Add(subjectOption);
        Options.Add(hoursOption);
        Options.Add(keyOption);
        Options.Add(quietOption);

        this.SetAction(parseResult =>
        {
            var role = parseResult.GetValue(roleOption);
            var subject = parseResult.GetValue(subjectOption)!;
            var hours = parseResult.GetValue(hoursOption);
            var key = parseResult.GetValue(keyOption) ?? DevJwtDefaults.Key;
            var quiet = parseResult.GetValue(quietOption);

            if (key.Length < 32)
            {
                AnsiConsole.MarkupLine("[red]X[/] Signing key must be at least 32 characters.");
                return 1;
            }

            var token = IssueToken(key, subject, role, TimeSpan.FromHours(hours));

            if (quiet)
            {
                Console.WriteLine(token);
                return 0;
            }

            AnsiConsole.MarkupLine("[green]JWT issued[/] " +
                $"(subject={Markup.Escape(subject)}, role={Markup.Escape(role ?? "<read-only>")}, expires in {hours}h)");
            AnsiConsole.WriteLine();
            Console.WriteLine(token);
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine("[grey]Use it as a bearer token:[/]");
            AnsiConsole.MarkupLine("[grey]  curl -H \"Authorization: Bearer <token>\" http://localhost:5000/info[/]");
            AnsiConsole.MarkupLine("[grey]  export MLOOP_TOKEN=$(mloop token -q)   # for clients reading MLOOP_TOKEN[/]");
            return 0;
        });
    }

    /// <summary>
    /// Builds an HS256 JWT with short claim names (<c>sub</c>/<c>role</c>) that the API consumes
    /// verbatim (it disables inbound claim mapping and reads role from the <c>role</c> claim).
    /// </summary>
    public static string IssueToken(string key, string subject, string? role, TimeSpan lifetime)
    {
        var claims = new List<Claim> { new("sub", subject) };
        if (!string.IsNullOrWhiteSpace(role))
            claims.Add(new Claim("role", role));

        var credentials = new SigningCredentials(
            new SymmetricSecurityKey(Encoding.UTF8.GetBytes(key)),
            SecurityAlgorithms.HmacSha256);

        var now = DateTime.UtcNow;
        var jwt = new JwtSecurityToken(
            issuer: DevJwtDefaults.Issuer,
            audience: DevJwtDefaults.Audience,
            claims: claims,
            notBefore: now,
            expires: now.Add(lifetime),
            signingCredentials: credentials);

        var handler = new JwtSecurityTokenHandler();
        handler.OutboundClaimTypeMap.Clear(); // keep short claim names ('sub'/'role'), no URI expansion
        return handler.WriteToken(jwt);
    }
}
