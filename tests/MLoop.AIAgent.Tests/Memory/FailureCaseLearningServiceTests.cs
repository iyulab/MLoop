using MemoryIndexer.Interfaces;
using MemoryIndexer.Models;
using MLoop.AIAgent.Core.Memory;
using MLoop.AIAgent.Core.Memory.Models;

namespace MLoop.AIAgent.Tests.Memory;

public class FailureCaseLearningServiceTests
{
    private readonly MockMemoryStore _mockStore;
    private readonly MockEmbeddingService _mockEmbedding;
    private readonly FailureCaseLearningService _service;

    public FailureCaseLearningServiceTests()
    {
        _mockStore = new MockMemoryStore();
        _mockEmbedding = new MockEmbeddingService();
        _service = new FailureCaseLearningService(_mockStore, _mockEmbedding);
    }

    #region Constructor Tests

    [Fact]
    public void Constructor_WithNullStore_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new FailureCaseLearningService(null!, _mockEmbedding));
    }

    [Fact]
    public void Constructor_WithNullEmbedding_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new FailureCaseLearningService(_mockStore, null!));
    }

    #endregion

    #region StoreFailureAsync Tests

    [Fact]
    public async Task StoreFailureAsync_WithNullContext_ThrowsArgumentNullException()
    {
        var resolution = CreateTestResolution();

        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            _service.StoreFailureAsync(null!, resolution));
    }

    [Fact]
    public async Task StoreFailureAsync_WithNullResolution_ThrowsArgumentNullException()
    {
        var context = CreateTestFailureContext();

        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            _service.StoreFailureAsync(context, null!));
    }

    [Fact]
    public async Task StoreFailureAsync_ValidInput_StoresMemory()
    {
        var context = CreateTestFailureContext();
        var resolution = CreateTestResolution();

        await _service.StoreFailureAsync(context, resolution);

        Assert.Single(_mockStore.StoredMemories);
        var stored = _mockStore.StoredMemories[0];
        Assert.Equal(MemoryType.Episodic, stored.Type);
        Assert.Equal("mloop:failures", stored.UserId);
        Assert.Equal(Tier.Long, stored.Tier);
    }

    [Fact]
    public async Task StoreFailureAsync_SetsCorrectMetadata()
    {
        var context = CreateTestFailureContext();
        var resolution = CreateTestResolution();

        await _service.StoreFailureAsync(context, resolution);

        var stored = _mockStore.StoredMemories[0];
        Assert.True(stored.TryGetMetadata<FailureContext>("context", out var storedContext));
        Assert.True(stored.TryGetMetadata<Resolution>("resolution", out var storedResolution));
        Assert.Equal(context.ErrorType, storedContext!.ErrorType);
        Assert.Equal(resolution.RootCause, storedResolution!.RootCause);
    }

    [Fact]
    public async Task StoreFailureAsync_VerifiedResolution_HasHigherImportance()
    {
        var context = CreateTestFailureContext();
        var verifiedResolution = CreateTestResolution();
        verifiedResolution.Verified = true;

        await _service.StoreFailureAsync(context, verifiedResolution);

        var stored = _mockStore.StoredMemories[0];
        Assert.True(stored.ImportanceScore > 0.5f);
    }

    #endregion

    #region CheckForSimilarFailuresAsync Tests

    [Fact]
    public async Task CheckForSimilarFailuresAsync_WithNullDatasetInfo_ThrowsArgumentNullException()
    {
        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            _service.CheckForSimilarFailuresAsync(null!));
    }

    [Fact]
    public async Task CheckForSimilarFailuresAsync_NoMatches_ReturnsEmptyList()
    {
        var datasetInfo = CreateTestDatasetInfo();

        var warnings = await _service.CheckForSimilarFailuresAsync(datasetInfo);

        Assert.Empty(warnings);
    }

    [Fact]
    public async Task CheckForSimilarFailuresAsync_WithMatches_ReturnsWarnings()
    {
        // Arrange - Store a failure first
        var context = CreateTestFailureContext();
        var resolution = CreateTestResolution();
        await _service.StoreFailureAsync(context, resolution);

        // Setup mock to return the stored failure
        _mockStore.SetupSearchResults([
            new MemorySearchResult
            {
                Memory = _mockStore.StoredMemories[0],
                Score = 0.75f
            }
        ]);

        var datasetInfo = CreateTestDatasetInfo();

        // Act
        var warnings = await _service.CheckForSimilarFailuresAsync(datasetInfo);

        // Assert
        Assert.Single(warnings);
        Assert.Equal(0.75f, warnings[0].SimilarityScore);
        Assert.Equal(WarningLevel.Medium, warnings[0].Level);
    }

    [Fact]
    public async Task CheckForSimilarFailuresAsync_HighScore_ReturnsHighWarningLevel()
    {
        var context = CreateTestFailureContext();
        var resolution = CreateTestResolution();
        await _service.StoreFailureAsync(context, resolution);

        _mockStore.SetupSearchResults([
            new MemorySearchResult
            {
                Memory = _mockStore.StoredMemories[0],
                Score = 0.95f
            }
        ]);

        var datasetInfo = CreateTestDatasetInfo();
        var warnings = await _service.CheckForSimilarFailuresAsync(datasetInfo);

        Assert.Single(warnings);
        Assert.Equal(WarningLevel.High, warnings[0].Level);
    }

    #endregion

    #region FindResolutionAsync Tests

    [Fact]
    public async Task FindResolutionAsync_WithEmptyErrorType_ThrowsArgumentException()
    {
        await Assert.ThrowsAsync<ArgumentException>(() =>
            _service.FindResolutionAsync("", "error message"));
    }

    [Fact]
    public async Task FindResolutionAsync_NoMatch_ReturnsNull()
    {
        var result = await _service.FindResolutionAsync(
            "InvalidDataException",
            "Test error message");

        Assert.Null(result);
    }

    [Fact]
    public async Task FindResolutionAsync_WithMatch_ReturnsSuggestion()
    {
        // Arrange
        var context = CreateTestFailureContext();
        var resolution = CreateTestResolution();
        await _service.StoreFailureAsync(context, resolution);

        _mockStore.SetupSearchResults([
            new MemorySearchResult
            {
                Memory = _mockStore.StoredMemories[0],
                Score = 0.85f
            }
        ]);

        // Act
        var result = await _service.FindResolutionAsync(
            "InvalidDataException",
            "Missing values in column");

        // Assert
        Assert.NotNull(result);
        Assert.Equal(0.85f, result.Confidence);
        Assert.NotNull(result.SuggestedResolution);
    }

    [Fact]
    public async Task FindResolutionAsync_ExactTypeMatch_SetsMatchedErrorType()
    {
        var context = CreateTestFailureContext();
        var resolution = CreateTestResolution();
        await _service.StoreFailureAsync(context, resolution);

        _mockStore.SetupSearchResults([
            new MemorySearchResult
            {
                Memory = _mockStore.StoredMemories[0],
                Score = 0.9f
            }
        ]);

        var result = await _service.FindResolutionAsync(
            "InvalidDataException", // Same as context.ErrorType
            "Any error message");

        Assert.NotNull(result);
        Assert.True(result.MatchedErrorType);
    }

    #endregion

    #region GetStatisticsAsync Tests

    [Fact]
    public async Task GetStatisticsAsync_EmptyStore_ReturnsZeroFailures()
    {
        var stats = await _service.GetStatisticsAsync();

        Assert.Equal(0, stats.TotalFailures);
    }

    [Fact]
    public async Task GetStatisticsAsync_WithFailures_ReturnsCorrectCount()
    {
        _mockStore.SetupTypeCounts(new Dictionary<MemoryType, int>
        {
            [MemoryType.Episodic] = 10
        });

        var stats = await _service.GetStatisticsAsync();

        Assert.Equal(10, stats.TotalFailures);
    }

    #endregion

    #region GetRecentFailuresAsync Tests

    [Fact]
    public async Task GetRecentFailuresAsync_EmptyStore_ReturnsEmptyList()
    {
        var failures = await _service.GetRecentFailuresAsync();

        Assert.Empty(failures);
    }

    [Fact]
    public async Task GetRecentFailuresAsync_WithFailures_ReturnsFailures()
    {
        var context = CreateTestFailureContext();
        var resolution = CreateTestResolution();
        await _service.StoreFailureAsync(context, resolution);

        _mockStore.SetupSearchResults([
            new MemorySearchResult
            {
                Memory = _mockStore.StoredMemories[0],
                Score = 0.5f
            }
        ]);

        var failures = await _service.GetRecentFailuresAsync();

        Assert.Single(failures);
        Assert.NotNull(failures[0].Context);
        Assert.NotNull(failures[0].Resolution);
    }

    [Fact]
    public async Task GetRecentFailuresAsync_WithPhaseFilter_FiltersResults()
    {
        // Store failure in "Training" phase
        var context = CreateTestFailureContext();
        context.Phase = "Training";
        var resolution = CreateTestResolution();
        await _service.StoreFailureAsync(context, resolution);

        _mockStore.SetupSearchResults([
            new MemorySearchResult
            {
                Memory = _mockStore.StoredMemories[0],
                Score = 0.5f
            }
        ]);

        // Query for "DataLoading" phase - should filter out Training phase
        var failures = await _service.GetRecentFailuresAsync(phase: "DataLoading");

        Assert.Empty(failures);
    }

    #endregion

    #region Helper Methods

    private static FailureContext CreateTestFailureContext()
    {
        return new FailureContext
        {
            ErrorType = "InvalidDataException",
            ErrorMessage = "Missing values detected in column 'Feature1'",
            Phase = "DataLoading",
            OccurredAt = DateTime.UtcNow,
            DatasetContext = new DatasetFingerprint
            {
                ColumnNames = ["Feature1", "Feature2", "Label"],
                RowCount = 1000,
                SizeCategory = "Medium"
            }
        };
    }

    private static Resolution CreateTestResolution()
    {
        return new Resolution
        {
            RootCause = "Data source had null values",
            FixDescription = "Applied MissingValueHandler with mean imputation",
            Changes = ["Added MissingValueHandler step to preprocessing"],
            PreventionAdvice = "Always validate data completeness before processing",
            Verified = false,
            ResolvedAt = DateTime.UtcNow
        };
    }

    private static DatasetInfo CreateTestDatasetInfo()
    {
        return new DatasetInfo
        {
            Fingerprint = new DatasetFingerprint
            {
                ColumnNames = ["Feature1", "Feature2", "Label"],
                RowCount = 1000,
                SizeCategory = "Medium"
            },
            CurrentPhase = "DataLoading",
            AppliedSteps = []
        };
    }

    #endregion
}

#region FailureWarning Tests

public class FailureWarningTests
{
    [Theory]
    [InlineData(0.95f, WarningLevel.High)]
    [InlineData(0.9f, WarningLevel.High)]
    [InlineData(0.85f, WarningLevel.Medium)]
    [InlineData(0.7f, WarningLevel.Medium)]
    [InlineData(0.5f, WarningLevel.Low)]
    [InlineData(0.3f, WarningLevel.Low)]
    public void Level_BasedOnSimilarityScore_ReturnsCorrectLevel(float score, WarningLevel expectedLevel)
    {
        var warning = new FailureWarning { SimilarityScore = score };

        Assert.Equal(expectedLevel, warning.Level);
    }

    [Fact]
    public void Message_ContainsRelevantInformation()
    {
        var warning = new FailureWarning
        {
            SimilarityScore = 0.85f,
            Context = new FailureContext
            {
                ErrorType = "TestError"
            },
            Resolution = new Resolution
            {
                PreventionAdvice = "Test prevention advice"
            }
        };

        var message = warning.Message;

        Assert.Contains("85%", message);
        Assert.Contains("TestError", message);
        Assert.Contains("Test prevention advice", message);
    }
}

#endregion

#region ResolutionSuggestion Tests

public class ResolutionSuggestionTests
{
    [Fact]
    public void Message_ContainsConfidenceAndResolutionInfo()
    {
        var suggestion = new ResolutionSuggestion
        {
            Confidence = 0.9f,
            SuggestedResolution = new Resolution
            {
                RootCause = "Test root cause",
                FixDescription = "Test fix description"
            }
        };

        var message = suggestion.Message;

        Assert.Contains("90%", message);
        Assert.Contains("Test root cause", message);
        Assert.Contains("Test fix description", message);
    }
}

#endregion
