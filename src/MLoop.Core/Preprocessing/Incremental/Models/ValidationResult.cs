namespace MLoop.Core.Preprocessing.Incremental.Models;

/// <summary>
/// Represents the result of a sampling validation.
/// </summary>
public sealed class ValidationResult
{
    /// <summary>
    /// Gets whether the validation passed.
    /// </summary>
    public required bool IsValid { get; init; }

    /// <summary>
    /// Gets the validation message (success or error description).
    /// </summary>
    public required string Message { get; init; }

    /// <summary>
    /// Gets the validation details.
    /// </summary>
    public Dictionary<string, object> Details { get; init; } = new();

    /// <summary>
    /// Creates a successful validation result.
    /// </summary>
    /// <param name="message">Success message.</param>
    /// <param name="details">Optional validation details.</param>
    /// <returns>A successful ValidationResult.</returns>
    public static ValidationResult Success(string message = "Validation passed", Dictionary<string, object>? details = null)
    {
        return new ValidationResult
        {
            IsValid = true,
            Message = message,
            Details = details ?? new Dictionary<string, object>()
        };
    }

    /// <summary>
    /// Creates a failed validation result.
    /// </summary>
    /// <param name="message">Failure message describing the issue.</param>
    /// <param name="details">Optional validation details.</param>
    /// <returns>A failed ValidationResult.</returns>
    public static ValidationResult Failure(string message, Dictionary<string, object>? details = null)
    {
        return new ValidationResult
        {
            IsValid = false,
            Message = message,
            Details = details ?? new Dictionary<string, object>()
        };
    }
}
