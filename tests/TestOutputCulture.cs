using System.Runtime.CompilerServices;

namespace MLoop.Tests.Shared;

/// <summary>
/// Pins the suite to the culture the product formats in, before any test runs.
/// </summary>
/// <remarks>
/// A test that asserts a rendered number is asserting the host's locale unless the culture is
/// fixed. That is not hypothetical: <c>{share:P1}</c> renders <c>97.5%</c> under en-US and ko-KR
/// but <c>97.5 %</c> under the invariant culture a CI runner starts in, which is how a suite can
/// be green on every developer machine and red on the runner that publishes the release. Pinning
/// here removes the difference in the only direction that is correct — towards what
/// <see cref="MLoop.Core.Globalization.OutputCulture"/> gives a user.
/// </remarks>
internal static class TestOutputCulture
{
    // CA2255 warns against ModuleInitializer in libraries, where an implicit global side effect can
    // surprise a consumer. A test assembly has no consumer but the runner, and pinning before the
    // first test is exactly the point.
#pragma warning disable CA2255
    [ModuleInitializer]
    internal static void Pin() => MLoop.Core.Globalization.OutputCulture.Pin();
#pragma warning restore CA2255
}
