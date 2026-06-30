using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;
using Microsoft.IdentityModel.Tokens;
using MLoop.CLI.Commands;
using MLoop.Core.Security;
using Xunit;

namespace MLoop.Tests.Commands;

/// <summary>
/// Pins that <c>mloop token</c> issues tokens the serve API accepts. The API validates with
/// MapInboundClaims=false, NameClaimType="sub", RoleClaimType="role" against
/// <see cref="DevJwtDefaults"/> — these tests mirror that exact configuration so issuer↔validator
/// drift is caught here rather than at runtime against a live server.
/// </summary>
public class TokenCommandTests
{
    private static TokenValidationParameters ApiValidationParameters(string key) => new()
    {
        ValidateIssuer = true,
        ValidateAudience = true,
        ValidateLifetime = true,
        ValidateIssuerSigningKey = true,
        ValidIssuer = DevJwtDefaults.Issuer,
        ValidAudience = DevJwtDefaults.Audience,
        IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(key)),
        NameClaimType = "sub",
        RoleClaimType = "role"
    };

    private static ClaimsPrincipal Validate(string token, string key)
    {
        var handler = new JwtSecurityTokenHandler { MapInboundClaims = false };
        return handler.ValidateToken(token, ApiValidationParameters(key), out _);
    }

    [Fact]
    public void IssuedReadOnlyToken_ValidatesAsAuthenticatedNonAdmin()
    {
        var token = TokenCommand.IssueToken(DevJwtDefaults.Key, "honeai-sim", role: null, TimeSpan.FromHours(1));

        var principal = Validate(token, DevJwtDefaults.Key);

        Assert.True(principal.Identity?.IsAuthenticated);
        Assert.Equal("honeai-sim", principal.Identity?.Name); // NameClaimType = "sub"
        Assert.False(principal.IsInRole("admin"));
    }

    [Fact]
    public void IssuedAdminToken_ValidatesWithAdminRole()
    {
        var token = TokenCommand.IssueToken(DevJwtDefaults.Key, "ops", role: "admin", TimeSpan.FromHours(1));

        var principal = Validate(token, DevJwtDefaults.Key);

        Assert.True(principal.Identity?.IsAuthenticated);
        Assert.True(principal.IsInRole("admin")); // RoleClaimType = "role"
    }

    [Fact]
    public void TokenSignedWithWrongKey_FailsValidation()
    {
        var token = TokenCommand.IssueToken(DevJwtDefaults.Key, "x", role: null, TimeSpan.FromHours(1));
        var wrongKey = new string('z', 40);

        Assert.ThrowsAny<SecurityTokenException>(() => Validate(token, wrongKey));
    }

    [Fact]
    public void CustomKey_RoundTripsUnderThatKey()
    {
        var customKey = "a-custom-server-signing-key-32-chars-min!!";
        var token = TokenCommand.IssueToken(customKey, "client", role: null, TimeSpan.FromHours(1));

        var principal = Validate(token, customKey);

        Assert.True(principal.Identity?.IsAuthenticated);
    }
}
