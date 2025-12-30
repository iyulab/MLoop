using Ironbees.Core.Guardrails;

namespace MLoop.AIAgent.Infrastructure;

/// <summary>
/// Factory class providing ML-specific guardrails for content validation.
/// Protects against PII leakage, injection attacks, and content policy violations.
/// </summary>
public static class MLGuardrails
{
    /// <summary>
    /// Creates a guardrail for detecting PII (Personally Identifiable Information).
    /// Detects email addresses, SSN, credit card numbers, phone numbers.
    /// </summary>
    /// <param name="validateInput">Whether to validate input content.</param>
    /// <param name="validateOutput">Whether to validate output content.</param>
    /// <returns>A configured RegexGuardrail for PII detection.</returns>
    public static RegexGuardrail CreatePIIGuardrail(bool validateInput = true, bool validateOutput = true)
    {
        return new RegexGuardrail(new RegexGuardrailOptions
        {
            Name = "PII-Detector",
            ValidateInput = validateInput,
            ValidateOutput = validateOutput,
            FindAllViolations = true,
            IncludeMatchedContent = false, // Don't log PII
            BlockedPatterns =
            [
                PatternDefinition.Create(
                    @"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                    "Email",
                    "Email address detected"),
                PatternDefinition.Create(
                    @"\b\d{3}-\d{2}-\d{4}\b",
                    "SSN",
                    "Social Security Number detected"),
                PatternDefinition.Create(
                    @"\b(?:\d{4}[-\s]?){3}\d{4}\b",
                    "CreditCard",
                    "Credit card number detected"),
                PatternDefinition.Create(
                    @"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
                    "PhoneNumber",
                    "Phone number detected"),
            ]
        });
    }

    /// <summary>
    /// Creates a guardrail for detecting injection attacks.
    /// Detects common SQL injection, command injection, and path traversal patterns.
    /// </summary>
    /// <param name="validateInput">Whether to validate input content.</param>
    /// <returns>A configured RegexGuardrail for injection detection.</returns>
    public static RegexGuardrail CreateInjectionGuardrail(bool validateInput = true)
    {
        return new RegexGuardrail(new RegexGuardrailOptions
        {
            Name = "Injection-Detector",
            ValidateInput = validateInput,
            ValidateOutput = false, // Only check input
            FindAllViolations = true,
            IncludeMatchedContent = true,
            BlockedPatterns =
            [
                PatternDefinition.Create(
                    @"(?i)(\bexec\s*\(|;\s*exec\b)",
                    "CommandInjection",
                    "Potential command injection detected"),
                PatternDefinition.Create(
                    @"(?i)(\bselect\b.*\bfrom\b|\binsert\b.*\binto\b|\bdelete\b.*\bfrom\b|\bupdate\b.*\bset\b|\bdrop\b.*\btable\b)",
                    "SQLInjection",
                    "Potential SQL injection detected"),
                PatternDefinition.Create(
                    @"(\.\.[\\/]|[\\/]\.\.)",
                    "PathTraversal",
                    "Path traversal attempt detected"),
            ]
        });
    }

    /// <summary>
    /// Creates a guardrail for content length validation.
    /// Prevents DoS via excessively long inputs.
    /// </summary>
    /// <param name="maxLength">Maximum allowed content length.</param>
    /// <param name="minLength">Minimum required content length.</param>
    /// <returns>A configured LengthGuardrail.</returns>
    public static LengthGuardrail CreateLengthGuardrail(int maxLength = 100000, int minLength = 1)
    {
        return new LengthGuardrail(new LengthGuardrailOptions
        {
            Name = "Length-Validator",
            MaxInputLength = maxLength,
            MinInputLength = minLength
        });
    }

    /// <summary>
    /// Creates a guardrail for blocking sensitive ML-related keywords.
    /// Prevents requests for sensitive operations.
    /// </summary>
    /// <param name="blockedKeywords">Optional list of additional blocked keywords.</param>
    /// <returns>A configured KeywordGuardrail.</returns>
    public static KeywordGuardrail CreateMLKeywordGuardrail(IEnumerable<string>? blockedKeywords = null)
    {
        var keywords = new List<string>
        {
            // Data exfiltration
            "extract all data",
            "dump database",
            "export credentials",
            // Model attacks
            "adversarial attack",
            "poison training",
            "backdoor model",
            // System access
            "system prompt",
            "ignore instructions",
            "jailbreak"
        };

        if (blockedKeywords != null)
        {
            keywords.AddRange(blockedKeywords);
        }

        return new KeywordGuardrail(new KeywordGuardrailOptions
        {
            Name = "ML-Keyword-Filter",
            BlockedKeywords = keywords,
            CaseSensitive = false,
            WholeWordOnly = false,
            ValidateInput = true,
            ValidateOutput = false,
            FindAllViolations = true
        });
    }

    /// <summary>
    /// Creates a standard ML guardrail pipeline with commonly needed protections.
    /// Includes: Length validation, PII detection, Injection detection.
    /// </summary>
    /// <param name="enablePII">Enable PII detection.</param>
    /// <param name="enableInjection">Enable injection detection.</param>
    /// <param name="enableLength">Enable length validation.</param>
    /// <param name="maxInputLength">Maximum input length.</param>
    /// <returns>A configured GuardrailPipeline.</returns>
    public static GuardrailPipeline CreateStandardPipeline(
        bool enablePII = true,
        bool enableInjection = true,
        bool enableLength = true,
        int maxInputLength = 100000)
    {
        var inputGuardrails = new List<IContentGuardrail>();
        var outputGuardrails = new List<IContentGuardrail>();

        if (enableLength)
        {
            inputGuardrails.Add(CreateLengthGuardrail(maxInputLength));
        }

        if (enableInjection)
        {
            inputGuardrails.Add(CreateInjectionGuardrail());
        }

        if (enablePII)
        {
            var piiGuardrail = CreatePIIGuardrail(validateInput: true, validateOutput: true);
            inputGuardrails.Add(piiGuardrail);
            outputGuardrails.Add(piiGuardrail);
        }

        return new GuardrailPipeline(
            inputGuardrails,
            outputGuardrails,
            new GuardrailPipelineOptions
            {
                FailFast = true,
                ThrowOnViolation = false,
                ThrowOnGuardrailError = false
            });
    }

    /// <summary>
    /// Creates a strict guardrail pipeline for production environments.
    /// Includes all protections with strict validation.
    /// </summary>
    /// <returns>A configured GuardrailPipeline with strict settings.</returns>
    public static GuardrailPipeline CreateStrictPipeline()
    {
        var inputGuardrails = new List<IContentGuardrail>
        {
            CreateLengthGuardrail(50000), // Stricter limit
            CreateInjectionGuardrail(),
            CreatePIIGuardrail(),
            CreateMLKeywordGuardrail()
        };

        var outputGuardrails = new List<IContentGuardrail>
        {
            CreatePIIGuardrail(validateInput: false, validateOutput: true)
        };

        return new GuardrailPipeline(
            inputGuardrails,
            outputGuardrails,
            new GuardrailPipelineOptions
            {
                FailFast = true,
                ThrowOnViolation = false,
                ThrowOnGuardrailError = false
            });
    }
}
