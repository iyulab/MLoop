namespace MLoop.AIAgent.Configuration;

/// <summary>
/// Configuration for memory-indexer integration.
/// </summary>
public class MemoryConfig
{
    /// <summary>
    /// Enable memory-indexer features (default: true for zero-config).
    /// </summary>
    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Storage directory for SQLite memory database.
    /// Default: .mloop/memory/
    /// </summary>
    public string StorageDirectory { get; set; } = Path.Combine(".mloop", "memory");

    /// <summary>
    /// SQLite database filename.
    /// Default: mloop-memory.db
    /// </summary>
    public string DatabaseName { get; set; } = "mloop-memory.db";

    /// <summary>
    /// Maximum database size in MB (auto-cleanup when exceeded).
    /// Default: 500MB
    /// </summary>
    public int MaxDatabaseSizeMB { get; set; } = 500;

    /// <summary>
    /// Maximum age of memories in days (auto-delete older entries).
    /// Default: 90 days
    /// </summary>
    public int MaxMemoryAgeDays { get; set; } = 90;

    /// <summary>
    /// Enable embedding for semantic search.
    /// Default: false (use local embeddings if available, otherwise disable)
    /// </summary>
    public bool EnableEmbedding { get; set; } = false;

    /// <summary>
    /// Embedding provider (Ollama, OpenAI, Local).
    /// Default: Ollama
    /// </summary>
    public string EmbeddingProvider { get; set; } = "Ollama";

    /// <summary>
    /// Embedding model name.
    /// Default: bge-m3
    /// </summary>
    public string EmbeddingModel { get; set; } = "bge-m3";

    /// <summary>
    /// Gets full database path.
    /// </summary>
    public string GetDatabasePath()
    {
        return Path.Combine(StorageDirectory, DatabaseName);
    }

    /// <summary>
    /// Creates zero-config default instance.
    /// </summary>
    public static MemoryConfig CreateDefault(string projectRoot)
    {
        return new MemoryConfig
        {
            Enabled = true,
            StorageDirectory = Path.Combine(projectRoot, ".mloop", "memory"),
            DatabaseName = "mloop-memory.db",
            MaxDatabaseSizeMB = 500,
            MaxMemoryAgeDays = 90,
            EnableEmbedding = false  // Start without embedding, enable when Ollama detected
        };
    }

    /// <summary>
    /// Validates configuration.
    /// </summary>
    public void Validate()
    {
        if (Enabled)
        {
            if (string.IsNullOrWhiteSpace(StorageDirectory))
            {
                throw new InvalidOperationException("Storage directory cannot be empty when memory is enabled.");
            }

            if (MaxDatabaseSizeMB <= 0)
            {
                throw new InvalidOperationException("Max database size must be positive.");
            }

            if (MaxMemoryAgeDays <= 0)
            {
                throw new InvalidOperationException("Max memory age must be positive.");
            }
        }
    }
}
