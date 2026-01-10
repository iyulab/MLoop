using MemoryIndexer;
using MemoryIndexer.Configuration;
using MemoryIndexer.Interfaces;
using MemoryIndexer.Sdk.Extensions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using MLoop.AIAgent.Configuration;

namespace MLoop.AIAgent.Infrastructure;

/// <summary>
/// Extension methods for registering memory-indexer services.
/// </summary>
public static class MemoryServiceExtensions
{
    /// <summary>
    /// Adds memory-indexer with zero-config defaults.
    /// </summary>
    /// <param name="services">Service collection.</param>
    /// <param name="projectRoot">Project root directory.</param>
    /// <param name="configure">Optional configuration.</param>
    /// <returns>Service collection for chaining.</returns>
    public static IServiceCollection AddMLoopMemory(
        this IServiceCollection services,
        string projectRoot,
        Action<MemoryConfig>? configure = null)
    {
        // Create and validate config
        var config = MemoryConfig.CreateDefault(projectRoot);
        configure?.Invoke(config);
        config.Validate();

        services.AddSingleton(config);

        if (!config.Enabled)
        {
            return services;
        }

        // Ensure storage directory exists
        Directory.CreateDirectory(config.StorageDirectory);

        // Add memory-indexer with simplified zero-config
        services.AddMemoryIndexer(options =>
        {
            // Storage: SQLite
            options.Storage.Type = StorageType.SqliteVec;
            options.Storage.ConnectionString = $"Data Source={config.GetDatabasePath()}";
            options.Storage.VectorDimensions = 384;  // Default dimensions

            // Embedding: Mock for zero-config (no external dependencies)
            if (config.EnableEmbedding)
            {
                options.Embedding.Provider = config.EmbeddingProvider switch
                {
                    "Ollama" => EmbeddingProvider.Ollama,
                    "OpenAI" => EmbeddingProvider.OpenAI,
                    "AzureOpenAI" => EmbeddingProvider.AzureOpenAI,
                    _ => EmbeddingProvider.Mock
                };
                options.Embedding.Model = config.EmbeddingModel;
                options.Embedding.Dimensions = 384;
            }
            else
            {
                options.Embedding.Provider = EmbeddingProvider.Mock;
                options.Embedding.Dimensions = 384;
            }

            // Search settings
            options.Search.DefaultLimit = 10;
        });

        return services;
    }

    /// <summary>
    /// Adds MemoryConversationService with memory integration.
    /// </summary>
    /// <param name="services">Service collection.</param>
    /// <param name="conversationsDirectory">Directory for conversation storage.</param>
    /// <param name="memoryEnabled">Enable memory features.</param>
    /// <returns>Service collection for chaining.</returns>
    public static IServiceCollection AddMemoryConversationService(
        this IServiceCollection services,
        string conversationsDirectory,
        bool memoryEnabled = true)
    {
        services.AddScoped<MemoryConversationService>(sp =>
        {
            var vcm = sp.GetRequiredService<IVirtualContextManager>();
            var buffer = sp.GetRequiredService<IBuffer>();
            var logger = sp.GetService<ILogger<MemoryConversationService>>();

            return new MemoryConversationService(
                vcm,
                buffer,
                conversationsDirectory,
                memoryEnabled,
                logger);
        });

        return services;
    }

    /// <summary>
    /// Detects if Ollama is available for local embedding.
    /// </summary>
    /// <returns>True if Ollama is accessible.</returns>
    public static async Task<bool> IsOllamaAvailableAsync()
    {
        try
        {
            using var httpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(2) };
            var response = await httpClient.GetAsync("http://localhost:11434/api/tags");
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Auto-detects and enables best embedding provider.
    /// </summary>
    /// <param name="config">Memory configuration.</param>
    /// <returns>Updated configuration with optimal embedding settings.</returns>
    public static async Task<MemoryConfig> AutoConfigureEmbeddingAsync(this MemoryConfig config)
    {
        // Check if Ollama is available
        if (await IsOllamaAvailableAsync())
        {
            config.EnableEmbedding = true;
            config.EmbeddingProvider = "Ollama";
            config.EmbeddingModel = "bge-m3";
            return config;
        }

        // Check if OpenAI API key is set
        var openAiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
        if (!string.IsNullOrWhiteSpace(openAiKey))
        {
            config.EnableEmbedding = true;
            config.EmbeddingProvider = "OpenAI";
            config.EmbeddingModel = "text-embedding-3-small";
            return config;
        }

        // Fallback: InMemory (no semantic search, but works)
        config.EnableEmbedding = false;
        return config;
    }
}
