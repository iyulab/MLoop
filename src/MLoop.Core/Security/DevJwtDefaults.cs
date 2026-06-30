namespace MLoop.Core.Security;

/// <summary>
/// Single authority for the local/development JWT parameters shared between the API
/// (which <em>validates</em> bearer tokens) and the CLI <c>mloop token</c> command
/// (which <em>issues</em> them). Keeping key/issuer/audience in one place prevents the
/// validator↔issuer drift that would otherwise make every issued token fail validation.
///
/// These are intentionally well-known defaults for frictionless local serving
/// (<c>mloop serve</c> on localhost). They are NOT secret and NOT for production: a real
/// deployment overrides <c>Jwt:Key</c> via configuration/environment and runs the API in
/// the Production environment, where the default key is rejected at startup.
/// </summary>
public static class DevJwtDefaults
{
    /// <summary>Well-known development signing key (≥32 chars, required by the validator).</summary>
    public const string Key = "MLoop-Default-Secret-Key-Change-In-Production-Min-32-Chars";

    /// <summary>Token issuer claim — must match the API's <c>ValidIssuer</c>.</summary>
    public const string Issuer = "MLoop.API";

    /// <summary>Token audience claim — must match the API's <c>ValidAudience</c>.</summary>
    public const string Audience = "MLoop.Client";
}
