using System.Reflection;
using MLoop.Core.AutoML;

namespace MLoop.Core.Tests.AutoML;

/// <summary>
/// MLoop.Core is a published library, and Microsoft.ML.AutoML is on its way out: no public member
/// may hand a caller an AutoML type, or the package cannot drop AutoML without breaking its users.
/// What still does is listed below, each with the step that removes it.
/// </summary>
public class AutoMLPublicSurfaceTests
{
    private const string AutoMLAssembly = "Microsoft.ML.AutoML";

    /// <summary>
    /// Public members that still expose an AutoML type, all of them column inference — replaced
    /// together with InferColumns, after which this list must be empty.
    /// </summary>
    private static readonly string[] StillExposed =
    [
        "CsvDataLoader.ApplyColumnPreservation",
        "CsvDataLoader.ReconcileInferredSchemaForInference",
    ];

    [Fact]
    public void NoPublicMember_ExposesAnAutoMLType_BeyondTheListedOnes()
    {
        var exposed = Exposures(typeof(AutoMLRunner).Assembly);

        Assert.Equal(StillExposed.Order(), exposed.Order());
    }

    // The check must be able to see an exposure: fed a type that has one of each kind, it finds all.

    [Fact]
    public void TheCheck_FindsEveryKindOfExposure()
    {
        var exposed = Exposures(typeof(Offender));

        Assert.Equal(
            new[]
            {
                $"{nameof(Offender)}.{nameof(Offender.Field)}",
                $"{nameof(Offender)}.{nameof(Offender.Property)}",
                $"{nameof(Offender)}.{nameof(Offender.Returns)}",
                $"{nameof(Offender)}.{nameof(Offender.Takes)}",
                $"{nameof(Offender)}.{nameof(Offender.Nested)}",
                $"{nameof(Offender)}..ctor",
            }.Order(),
            exposed.Order());
    }

    [Fact]
    public void TheCheck_IgnoresNonPublicMembers()
    {
        Assert.Empty(Exposures(typeof(Hidden)));
    }

    public sealed class Offender
    {
        public Microsoft.ML.AutoML.ColumnInformation? Field;
        public Offender(Microsoft.ML.AutoML.ColumnInformation info) => Field = info;
        public Microsoft.ML.AutoML.ColumnInformation? Property { get; set; }
        public Microsoft.ML.AutoML.ColumnInformation? Returns() => null;
        public void Takes(Microsoft.ML.AutoML.ColumnInformation info) { }
        public List<Microsoft.ML.AutoML.ColumnInformation>? Nested() => null;
        public int Fine() => 0;
    }

    public sealed class Hidden
    {
        internal Microsoft.ML.AutoML.ColumnInformation? Internal() => null;
        private Microsoft.ML.AutoML.ColumnInformation? _private;
        public bool HasInfo => _private is not null;
    }

    private static List<string> Exposures(Assembly assembly) =>
        [.. assembly.GetExportedTypes().SelectMany(Exposures)];

    private static IEnumerable<string> Exposures(Type type)
    {
        if (type.BaseType is { } baseType && IsAutoML(baseType))
            yield return $"{type.Name} : {baseType.Name}";
        foreach (var iface in type.GetInterfaces().Where(IsAutoML))
            yield return $"{type.Name} : {iface.Name}";

        const BindingFlags visible = BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly;
        foreach (var member in type.GetMembers(visible))
        {
            var types = member switch
            {
                FieldInfo f => [f.FieldType],
                PropertyInfo p => [p.PropertyType],
                MethodInfo m when !m.IsSpecialName => m.GetParameters().Select(x => x.ParameterType).Append(m.ReturnType),
                ConstructorInfo c => c.GetParameters().Select(x => x.ParameterType),
                EventInfo e => [e.EventHandlerType!],
                _ => Enumerable.Empty<Type>()
            };
            if (types.Any(IsAutoML))
                yield return $"{type.Name}.{member.Name}";
        }
    }

    private static bool IsAutoML(Type type)
    {
        if (type.HasElementType)
            return IsAutoML(type.GetElementType()!);
        if (type.Assembly.GetName().Name == AutoMLAssembly)
            return true;
        return type.IsGenericType && type.GetGenericArguments().Any(IsAutoML);
    }
}
