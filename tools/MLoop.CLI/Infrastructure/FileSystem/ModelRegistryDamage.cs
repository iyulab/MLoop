using MLoop.Core.Storage;

namespace MLoop.CLI.Infrastructure.FileSystem;

/// <summary>
/// The production model for a caller that is deciding whether to replace it. <see cref="IModelRegistry.GetProductionAsync"/>
/// throws <see cref="UnreadableStateFileException"/> when the pointer exists and cannot be read - damage is not "no production
/// model" - but the way back it names is <c>mloop promote</c>, and a promotion that stopped on the same unreadable pointer
/// would send the user in a circle. A promotion goes on without a baseline and says why.
/// </summary>
public static class ModelRegistryDamage
{
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, byte> Reported = new();

    public static async Task<ModelInfo?> GetProductionOrNoneAsync(
        this IModelRegistry registry, string modelName, CancellationToken cancellationToken = default)
    {
        try
        {
            return await registry.GetProductionAsync(modelName, cancellationToken);
        }
        catch (UnreadableStateFileException damage)
        {
            if (Reported.TryAdd(damage.Path, 0))
                Console.Error.WriteLine($"Warning: {damage.Path} cannot be read ({damage.Reason}); going on without a production model to compare with.");
            return null;
        }
    }
}
