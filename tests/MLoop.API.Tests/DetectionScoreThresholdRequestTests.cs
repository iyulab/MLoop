using System.Text.Json;
using Microsoft.AspNetCore.Http;
using MLoop.API;
using MLoop.Core.Prediction;

namespace MLoop.API.Tests;

/// <summary>
/// <c>POST /predict</c> for a detector takes the confidence a box needs from <c>?scoreThreshold=</c> or the JSON body, and refuses a
/// value that cannot select a box instead of ignoring it.
/// </summary>
public class DetectionScoreThresholdRequestTests
{
    private static HttpRequest Request(string? query)
    {
        var context = new DefaultHttpContext();
        if (query is not null)
            context.Request.QueryString = new QueryString(query);
        return context.Request;
    }

    private static JsonElement Body(string json) => JsonDocument.Parse(json).RootElement.Clone();

    [Fact]
    public void NothingGiven_IsTheDefault()
    {
        var (threshold, error) = ObjectDetectionPredictions.ReadScoreThreshold(Request(null), Body("{\"images\":[]}"));

        Assert.Null(error);
        Assert.Equal(DetectionScoreThreshold.Default, threshold);
    }

    [Fact]
    public void TheBodyField_IsRead()
    {
        var (threshold, error) = ObjectDetectionPredictions.ReadScoreThreshold(Request(null), Body("{\"images\":[],\"scoreThreshold\":0.1}"));

        Assert.Null(error);
        Assert.Equal(0.1, threshold);
    }

    [Fact]
    public void TheQuery_WinsOverTheBody()
    {
        var (threshold, error) = ObjectDetectionPredictions.ReadScoreThreshold(Request("?scoreThreshold=0.3"), Body("{\"scoreThreshold\":0.1}"));

        Assert.Null(error);
        Assert.Equal(0.3, threshold);
    }

    [Theory]
    [InlineData("?scoreThreshold=0")]
    [InlineData("?scoreThreshold=1.5")]
    [InlineData("?scoreThreshold=-1")]
    [InlineData("?scoreThreshold=high")]
    public void ANumberThatSelectsNothingOrNoNumber_IsRefusedWithWhatWasGiven(string query)
    {
        var (_, error) = ObjectDetectionPredictions.ReadScoreThreshold(Request(query), Body("{}"));

        Assert.NotNull(error);
        Assert.Contains(query.Split('=')[1], error, StringComparison.Ordinal);
    }

    [Fact]
    public void ABodyThatIsAnArray_FallsBackToTheDefault()
    {
        var (threshold, error) = ObjectDetectionPredictions.ReadScoreThreshold(Request(null), Body("[1,2]"));

        Assert.Null(error);
        Assert.Equal(DetectionScoreThreshold.Default, threshold);
    }
}
