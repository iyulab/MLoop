using System.Text.Json;
using MLoop.DataStore.Services;

namespace MLoop.DataStore.Tests;

/// <summary>
/// Pins the single feedback predicted-vs-actual matching rule (F-29). The CLI's "Match" column and
/// the accuracy calculation previously had separate copies that drifted — the CLI's naive copy
/// compared raw <see cref="JsonElement"/> structs, so the list always showed a mismatch even when
/// the metrics command reported a match. These tests exercise the JSON-sourced types that actually
/// flow at runtime, not just CLR primitives.
/// </summary>
public class FeedbackValueComparerTests
{
    // At runtime, feedback values round-trip through JSON as JsonElement (STJ deserializes object).
    private static JsonElement Json(string raw) => JsonDocument.Parse(raw).RootElement.Clone();

    [Fact]
    public void BothNull_ReturnsTrue() => Assert.True(FeedbackValueComparer.ValuesMatch(null, null));

    [Fact]
    public void OneNull_ReturnsFalse()
    {
        Assert.False(FeedbackValueComparer.ValuesMatch("a", null));
        Assert.False(FeedbackValueComparer.ValuesMatch(null, "a"));
    }

    [Fact]
    public void SameString_ReturnsTrue() => Assert.True(FeedbackValueComparer.ValuesMatch("OK", "OK"));

    [Fact]
    public void CaseInsensitiveString_ReturnsTrue()
    {
        Assert.True(FeedbackValueComparer.ValuesMatch("ok", "OK"));
        Assert.True(FeedbackValueComparer.ValuesMatch("Good", "GOOD"));
    }

    [Fact]
    public void DifferentStrings_ReturnsFalse() => Assert.False(FeedbackValueComparer.ValuesMatch("OK", "NG"));

    [Fact]
    public void EqualIntegers_ReturnsTrue() => Assert.True(FeedbackValueComparer.ValuesMatch(42, 42));

    [Fact]
    public void DifferentIntegers_ReturnsFalse() => Assert.False(FeedbackValueComparer.ValuesMatch(42, 43));

    [Fact]
    public void EqualDoubles_ReturnsTrue() => Assert.True(FeedbackValueComparer.ValuesMatch(3.14, 3.14));

    // --- F-29 regression: the runtime JsonElement cases the old naive CLI copy got wrong ---

    [Fact]
    public void JsonElementEqualStrings_ReturnsTrue()
        => Assert.True(FeedbackValueComparer.ValuesMatch(Json("\"OK\""), Json("\"OK\"")));

    [Fact]
    public void JsonElementCaseInsensitiveStrings_ReturnsTrue()
        => Assert.True(FeedbackValueComparer.ValuesMatch(Json("\"ok\""), Json("\"OK\"")));

    [Fact]
    public void JsonElementDifferentStrings_ReturnsFalse()
        => Assert.False(FeedbackValueComparer.ValuesMatch(Json("\"OK\""), Json("\"NG\"")));

    [Fact]
    public void JsonElementEqualNumbers_ReturnsTrue()
        => Assert.True(FeedbackValueComparer.ValuesMatch(Json("42"), Json("42")));

    [Fact]
    public void JsonElementEqualDoublesWithinTolerance_ReturnsTrue()
        => Assert.True(FeedbackValueComparer.ValuesMatch(Json("3.14159"), Json("3.14159")));

    [Fact]
    public void JsonElementDifferentNumbers_ReturnsFalse()
        => Assert.False(FeedbackValueComparer.ValuesMatch(Json("42"), Json("43")));

    // --- F-30: numeric prediction vs string actual (the CLI actual is always a string) ---

    [Fact]
    public void JsonNumberPredicted_StringActual_SameValue_ReturnsTrue()
        => Assert.True(FeedbackValueComparer.ValuesMatch(Json("1"), Json("\"1\"")));

    [Fact]
    public void JsonNumberPredicted_StringActual_DifferentValue_ReturnsFalse()
        => Assert.False(FeedbackValueComparer.ValuesMatch(Json("1"), Json("\"2\"")));

    [Fact]
    public void JsonDoublePredicted_StringActual_SameValue_ReturnsTrue()
        => Assert.True(FeedbackValueComparer.ValuesMatch(Json("3.14"), Json("\"3.14\"")));

    [Fact]
    public void NumericStrings_SameValueDifferentForm_ReturnsTrue()
        // "1.0" and "1" are not both pure labels here — one side numeric → numeric compare.
        => Assert.True(FeedbackValueComparer.ValuesMatch(1.0, "1"));

    [Fact]
    public void BooleanPredicted_StringActual_ReturnsTrue()
        => Assert.True(FeedbackValueComparer.ValuesMatch(Json("true"), Json("\"true\"")));

    [Fact]
    public void NonNumericStringLabels_StayCaseInsensitive_AndDistinct()
    {
        Assert.True(FeedbackValueComparer.ValuesMatch(Json("\"OK\""), Json("\"ok\"")));
        Assert.False(FeedbackValueComparer.ValuesMatch(Json("\"01\""), Json("\"1\"")));
    }
}
