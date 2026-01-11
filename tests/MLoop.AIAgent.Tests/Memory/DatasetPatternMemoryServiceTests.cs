using MemoryIndexer.Interfaces;
using MemoryIndexer.Models;
using MLoop.AIAgent.Core.Memory;
using MLoop.AIAgent.Core.Memory.Models;

namespace MLoop.AIAgent.Tests.Memory;

public class DatasetPatternMemoryServiceTests
{
    private readonly MockMemoryStore _mockStore;
    private readonly MockEmbeddingService _mockEmbedding;
    private readonly DatasetPatternMemoryService _service;

    public DatasetPatternMemoryServiceTests()
    {
        _mockStore = new MockMemoryStore();
        _mockEmbedding = new MockEmbeddingService();
        _service = new DatasetPatternMemoryService(_mockStore, _mockEmbedding);
    }

    #region Constructor Tests

    [Fact]
    public void Constructor_WithNullStore_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new DatasetPatternMemoryService(null!, _mockEmbedding));
    }

    [Fact]
    public void Constructor_WithNullEmbedding_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new DatasetPatternMemoryService(_mockStore, null!));
    }

    #endregion

    #region StorePatternAsync Tests

    [Fact]
    public async Task StorePatternAsync_WithNullFingerprint_ThrowsArgumentNullException()
    {
        var outcome = new ProcessingOutcome { Success = true };

        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            _service.StorePatternAsync(null!, outcome));
    }

    [Fact]
    public async Task StorePatternAsync_WithNullOutcome_ThrowsArgumentNullException()
    {
        var fingerprint = CreateTestFingerprint();

        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            _service.StorePatternAsync(fingerprint, null!));
    }

    [Fact]
    public async Task StorePatternAsync_WithFailedOutcome_DoesNotStore()
    {
        var fingerprint = CreateTestFingerprint();
        var outcome = new ProcessingOutcome { Success = false, ErrorMessage = "Test error" };

        await _service.StorePatternAsync(fingerprint, outcome);

        Assert.Empty(_mockStore.StoredMemories);
    }

    [Fact]
    public async Task StorePatternAsync_WithSuccessfulOutcome_StoresMemory()
    {
        var fingerprint = CreateTestFingerprint();
        var outcome = CreateTestOutcome();

        await _service.StorePatternAsync(fingerprint, outcome);

        Assert.Single(_mockStore.StoredMemories);
        var stored = _mockStore.StoredMemories[0];
        Assert.Equal(MemoryType.Procedural, stored.Type);
        Assert.Equal("mloop:patterns", stored.UserId);
        Assert.Equal(Tier.Long, stored.Tier);
    }

    [Fact]
    public async Task StorePatternAsync_SetsCorrectMetadata()
    {
        var fingerprint = CreateTestFingerprint();
        var outcome = CreateTestOutcome();

        await _service.StorePatternAsync(fingerprint, outcome);

        var stored = _mockStore.StoredMemories[0];
        Assert.True(stored.TryGetMetadata<DatasetFingerprint>("fingerprint", out var storedFingerprint));
        Assert.True(stored.TryGetMetadata<ProcessingOutcome>("outcome", out var storedOutcome));
        Assert.Equal(fingerprint.Hash, storedFingerprint!.Hash);
        Assert.Equal(outcome.Success, storedOutcome!.Success);
    }

    #endregion

    #region FindSimilarPatternsAsync Tests

    [Fact]
    public async Task FindSimilarPatternsAsync_WithNullFingerprint_ThrowsArgumentNullException()
    {
        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            _service.FindSimilarPatternsAsync(null!));
    }

    [Fact]
    public async Task FindSimilarPatternsAsync_NoResults_ReturnsEmptyList()
    {
        var fingerprint = CreateTestFingerprint();

        var results = await _service.FindSimilarPatternsAsync(fingerprint);

        Assert.Empty(results);
    }

    [Fact]
    public async Task FindSimilarPatternsAsync_WithStoredPatterns_ReturnsRecommendations()
    {
        // Arrange - Store a pattern first
        var fingerprint = CreateTestFingerprint();
        var outcome = CreateTestOutcome();
        await _service.StorePatternAsync(fingerprint, outcome);

        // Setup mock to return the stored pattern
        _mockStore.SetupSearchResults([
            new MemorySearchResult
            {
                Memory = _mockStore.StoredMemories[0],
                Score = 0.85f
            }
        ]);

        // Act
        var results = await _service.FindSimilarPatternsAsync(fingerprint);

        // Assert
        Assert.Single(results);
        Assert.Equal(0.85f, results[0].SimilarityScore);
    }

    #endregion

    #region GetRecommendationAsync Tests

    [Fact]
    public async Task GetRecommendationAsync_NoMatch_ReturnsNull()
    {
        var fingerprint = CreateTestFingerprint();

        var result = await _service.GetRecommendationAsync(fingerprint);

        Assert.Null(result);
    }

    [Fact]
    public async Task GetRecommendationAsync_WithMatch_ReturnsTopResult()
    {
        // Arrange
        var fingerprint = CreateTestFingerprint();
        var outcome = CreateTestOutcome();
        await _service.StorePatternAsync(fingerprint, outcome);

        _mockStore.SetupSearchResults([
            new MemorySearchResult
            {
                Memory = _mockStore.StoredMemories[0],
                Score = 0.9f
            }
        ]);

        // Act
        var result = await _service.GetRecommendationAsync(fingerprint, minConfidence: 0.7f);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(0.9f, result.Confidence);
    }

    #endregion

    #region GetStatisticsAsync Tests

    [Fact]
    public async Task GetStatisticsAsync_EmptyStore_ReturnsZeroPatterns()
    {
        var stats = await _service.GetStatisticsAsync();

        Assert.Equal(0, stats.TotalPatterns);
    }

    [Fact]
    public async Task GetStatisticsAsync_WithPatterns_ReturnsCorrectCount()
    {
        // Setup mock to return count
        _mockStore.SetupTypeCounts(new Dictionary<MemoryType, int>
        {
            [MemoryType.Procedural] = 5
        });

        var stats = await _service.GetStatisticsAsync();

        Assert.Equal(5, stats.TotalPatterns);
    }

    #endregion

    #region Helper Methods

    private static DatasetFingerprint CreateTestFingerprint()
    {
        return new DatasetFingerprint
        {
            ColumnNames = ["Feature1", "Feature2", "Label"],
            ColumnTypes = new Dictionary<string, string>
            {
                ["Feature1"] = "Double",
                ["Feature2"] = "String",
                ["Label"] = "String"
            },
            RowCount = 1000,
            SizeCategory = "Medium",
            NumericRatio = 0.33,
            CategoricalRatio = 0.67,
            Hash = "ABCD1234"
        };
    }

    private static ProcessingOutcome CreateTestOutcome()
    {
        return new ProcessingOutcome
        {
            Success = true,
            Steps =
            [
                new PreprocessingStep { Type = "MissingValueHandler", Order = 1 },
                new PreprocessingStep { Type = "OneHotEncoder", Order = 2 }
            ],
            ProcessingTimeMs = 1500,
            PerformanceMetrics = new Dictionary<string, double>
            {
                ["Accuracy"] = 0.92
            },
            BestTrainer = "LightGbm"
        };
    }

    #endregion
}

#region Mock Implementations

internal class MockMemoryStore : IMemoryStore
{
    public List<MemoryUnit> StoredMemories { get; } = [];
    private IReadOnlyList<MemorySearchResult> _searchResults = [];
    private IReadOnlyDictionary<MemoryType, int> _typeCounts = new Dictionary<MemoryType, int>();

    public void SetupSearchResults(List<MemorySearchResult> results) => _searchResults = results;
    public void SetupTypeCounts(Dictionary<MemoryType, int> counts) => _typeCounts = counts;

    public Task<MemoryUnit> StoreAsync(MemoryUnit memory, CancellationToken cancellationToken = default)
    {
        memory.Id = Guid.NewGuid();
        StoredMemories.Add(memory);
        return Task.FromResult(memory);
    }

    public Task<IReadOnlyList<MemorySearchResult>> SearchAsync(
        ReadOnlyMemory<float> queryEmbedding,
        MemorySearchOptions options,
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult(_searchResults);
    }

    public Task<IReadOnlyDictionary<MemoryType, int>> GetTypeCountsAsync(
        string userId,
        CancellationToken cancellationToken = default)
    {
        return Task.FromResult(_typeCounts);
    }

    // Required interface methods - minimal implementation
    public Task<IReadOnlyList<MemoryUnit>> StoreBatchAsync(IEnumerable<MemoryUnit> memories, CancellationToken cancellationToken = default)
        => Task.FromResult<IReadOnlyList<MemoryUnit>>([]);
    public Task<MemoryUnit?> GetByIdAsync(Guid id, CancellationToken cancellationToken = default)
        => Task.FromResult<MemoryUnit?>(null);
    public Task<IReadOnlyList<MemoryUnit>> GetByIdsAsync(IEnumerable<Guid> ids, CancellationToken cancellationToken = default)
        => Task.FromResult<IReadOnlyList<MemoryUnit>>([]);
    public Task<bool> UpdateAsync(MemoryUnit memory, CancellationToken cancellationToken = default)
        => Task.FromResult(false);
    public Task<bool> DeleteAsync(Guid id, bool hardDelete = false, CancellationToken cancellationToken = default)
        => Task.FromResult(false);
    public Task<IReadOnlyList<MemoryUnit>> GetAllAsync(string userId, MemoryFilterOptions? options = null, CancellationToken cancellationToken = default)
        => Task.FromResult<IReadOnlyList<MemoryUnit>>([]);
    public Task<long> GetCountAsync(string userId, CancellationToken cancellationToken = default)
        => Task.FromResult(0L);
    public Task EnsureCollectionExistsAsync(CancellationToken cancellationToken = default)
        => Task.CompletedTask;
    public Task DeleteCollectionAsync(CancellationToken cancellationToken = default)
        => Task.CompletedTask;
}

internal class MockEmbeddingService : IEmbeddingService
{
    public int Dimensions => 384;

    public Task<ReadOnlyMemory<float>> GenerateEmbeddingAsync(string text, CancellationToken cancellationToken = default)
    {
        // Return a simple mock embedding
        return Task.FromResult(new ReadOnlyMemory<float>(new float[384]));
    }

    public Task<IReadOnlyList<ReadOnlyMemory<float>>> GenerateBatchEmbeddingsAsync(IEnumerable<string> texts, CancellationToken cancellationToken = default)
    {
        return Task.FromResult<IReadOnlyList<ReadOnlyMemory<float>>>(
            texts.Select(_ => new ReadOnlyMemory<float>(new float[384])).ToList());
    }
}

#endregion
