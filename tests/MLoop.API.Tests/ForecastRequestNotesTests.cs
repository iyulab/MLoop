using System.Text.Json;
using FluentAssertions;
using MLoop.API;

namespace MLoop.API.Tests;

/// <summary>
/// A forecast replays the series kept beside the model, so the request can set only <c>horizon</c>. Anything else in the body is
/// said to be unused rather than answered as if it had been the series to forecast. Pure — no native library needed.
/// </summary>
public sealed class ForecastRequestNotesTests
{
    private static JsonElement Json(string text) => JsonDocument.Parse(text).RootElement.Clone();

    [Theory]
    [InlineData("{}")]
    [InlineData("{\"horizon\":5}")]
    [InlineData("[]")]
    [InlineData("null")]
    public void A_body_that_holds_nothing_the_forecast_ignores_has_no_note(string body) =>
        ForecastRequestNotes.Of(Json(body)).Should().BeEmpty();

    [Fact]
    public void A_row_is_said_to_be_unused_and_named()
    {
        var notes = ForecastRequestNotes.Of(Json("{\"value\":1.5}"));

        var note = notes.Should().ContainSingle().Subject;
        note.Should().Contain("\"value\"").And.Contain("not used").And.Contain("series kept with the model");
    }

    [Fact]
    public void Rows_are_said_to_be_unused_and_counted()
    {
        var notes = ForecastRequestNotes.Of(Json("[{\"value\":1},{\"value\":2}]"));

        notes.Should().ContainSingle().Which.Should().Contain("2 rows").And.Contain("not used");
    }

    [Fact]
    public void A_horizon_that_is_not_a_number_is_said_to_be_unused()
    {
        var notes = ForecastRequestNotes.Of(Json("{\"horizon\":\"10\"}"));

        notes.Should().ContainSingle().Which.Should().Contain("\"horizon\" must be a number");
    }

    [Fact]
    public void A_valid_horizon_beside_other_fields_still_notes_the_other_fields()
    {
        var notes = ForecastRequestNotes.Of(Json("{\"horizon\":5,\"a\":1,\"b\":2,\"c\":3,\"d\":4,\"e\":5,\"f\":6}"));

        notes.Should().ContainSingle().Which.Should().Contain("\"a\", \"b\", \"c\", \"d\", \"e\" and 1 more").And.NotContain("\"f\"");
    }
}
