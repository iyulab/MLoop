using System.Runtime.CompilerServices;
using MLoop.Core.AutoML;

namespace MLoop.Core.DeepLearning.Tests;

/// <summary>
/// Registers the deep-learning module for this assembly, before any test runs.
/// </summary>
/// <remarks>
/// <para>
/// <c>DataLoaderFactory</c> resolves an image or object-detection loader only when the module has
/// been registered — in production that is the application's job, done once at the entry point. For
/// a test assembly whose whole subject is that module, the equivalent of an entry point is this.
/// </para>
/// <para>
/// It used to live in one test class's static constructor, which runs when that class is first
/// touched. A second class needed the same registration and did not have one, so whether it passed
/// depended on which class the runner reached first — green while the assembly ran its classes in
/// parallel, and a <c>NotSupportedException</c> the moment it did not. A precondition of the
/// assembly belongs to the assembly.
/// </para>
/// </remarks>
internal static class DeepLearningModuleRegistration
{
    // CA2255 warns against ModuleInitializer in libraries, where a hidden global side effect can
    // surprise a consumer. This assembly's only consumer is the test runner, and the side effect is
    // exactly what production performs at startup.
#pragma warning disable CA2255
    [ModuleInitializer]
    internal static void Register() => DeepLearningRegistry.Register(new DeepLearningModule());
#pragma warning restore CA2255
}
