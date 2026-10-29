using System.Collections;
using System.Reflection;
using MLoop.Core.Text;
using YamlDotNet.RepresentationModel;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace MLoop.CLI.Infrastructure.Configuration;

/// <summary>A key in <c>mloop.yaml</c> that nothing reads, where it stands, and the nearest key that something does.</summary>
public sealed record UnknownConfigKey(string Path, string? Suggestion)
{
    /// <summary>The sentence <c>validate</c> and the commands that load the file say.</summary>
    public string Describe() => Suggestion is null
        ? $"'{Path}' is not a setting mloop reads — it is ignored"
        : $"'{Path}' is not a setting mloop reads — it is ignored. Did you mean '{Suggestion}'?";
}

/// <summary>
/// Finds the keys of a <c>mloop.yaml</c> that the configuration types have no property for.
/// </summary>
/// <remarks>
/// <para>
/// The file is read leniently on purpose — an older mloop meeting a key a newer one added must not stop —
/// but lenient used to mean silent, and for most settings silence is only a setting that did not take
/// effect. For the promotion gate it was worse: a misspelled <c>max</c> left a class limit that refused every
/// model, and a misspelled <c>no_worse_than_production</c> left a rule that was never judged. The file is
/// still read the same way; this says what was dropped.
/// </para>
/// <para>
/// The walk compares the YAML tree with the types the file is read into, so there is no second list of
/// settings to keep in step with them: a property added to a configuration type is known at once. The keys
/// of a dictionary (model names, column names) are the user's own and are never reported; only their values
/// are walked.
/// </para>
/// </remarks>
public static class ConfigKeyAudit
{
    private static readonly INamingConvention Convention = UnderscoredNamingConvention.Instance;

    /// <summary>The unknown keys of <paramref name="yaml"/> read as <typeparamref name="TRoot"/>, in file order.</summary>
    public static IReadOnlyList<UnknownConfigKey> Find<TRoot>(string yaml) => Find(yaml, typeof(TRoot));

    public static IReadOnlyList<UnknownConfigKey> Find(string yaml, Type root)
    {
        var stream = new YamlStream();
        try
        {
            stream.Load(new StringReader(yaml));
        }
        catch (YamlDotNet.Core.YamlException)
        {
            return [];   // a file that does not parse is reported where it is parsed
        }

        var found = new List<UnknownConfigKey>();
        if (stream.Documents.Count > 0)
            Walk(stream.Documents[0].RootNode, root, "", found);
        return found;
    }

    private static void Walk(YamlNode node, Type type, string path, List<UnknownConfigKey> found)
    {
        type = Nullable.GetUnderlyingType(type) ?? type;

        if (node is YamlMappingNode map && DictionaryValueType(type) is { } valueType)
        {
            foreach (var (key, value) in map.Children)
                Walk(value, valueType, Join(path, Scalar(key)), found);
        }
        else if (node is YamlSequenceNode sequence && ElementType(type) is { } elementType)
        {
            for (var i = 0; i < sequence.Children.Count; i++)
                Walk(sequence.Children[i], elementType, $"{path}[{i}]", found);
        }
        else if (node is YamlMappingNode members && IsConfigurationClass(type))
        {
            var properties = Readable(type);
            foreach (var (keyNode, value) in members.Children)
            {
                var key = Scalar(keyNode);
                if (properties.TryGetValue(key, out var property))
                    Walk(value, property.PropertyType, Join(path, key), found);
                else
                    found.Add(new UnknownConfigKey(Join(path, key), EditDistance.Nearest(key, properties.Keys)));
            }
        }
    }

    /// <summary>The keys a type reads: its public settable properties, spelled as the file spells them.</summary>
    private static Dictionary<string, PropertyInfo> Readable(Type type)
    {
        var keys = new Dictionary<string, PropertyInfo>(StringComparer.Ordinal);
        foreach (var property in type.GetProperties(BindingFlags.Public | BindingFlags.Instance))
        {
            if (property.GetSetMethod() is null || property.GetIndexParameters().Length > 0
                || property.GetCustomAttribute<YamlIgnoreAttribute>() is not null)
                continue;
            var alias = property.GetCustomAttribute<YamlMemberAttribute>()?.Alias;
            keys[string.IsNullOrEmpty(alias) ? Convention.Apply(property.Name) : alias] = property;
        }
        return keys;
    }

    private static bool IsConfigurationClass(Type type) =>
        type.IsClass && type != typeof(string) && type != typeof(object)
        && !typeof(IEnumerable).IsAssignableFrom(type);

    private static Type? DictionaryValueType(Type type)
    {
        foreach (var candidate in type.GetInterfaces().Append(type))
        {
            if (candidate.IsGenericType && candidate.GetGenericTypeDefinition() == typeof(IDictionary<,>))
                return candidate.GetGenericArguments()[1];
        }
        return null;
    }

    private static Type? ElementType(Type type)
    {
        if (type == typeof(string) || DictionaryValueType(type) is not null)
            return null;
        foreach (var candidate in type.GetInterfaces().Append(type))
        {
            if (candidate.IsGenericType && candidate.GetGenericTypeDefinition() == typeof(IEnumerable<>))
                return candidate.GetGenericArguments()[0];
        }
        return null;
    }

    private static string Scalar(YamlNode node) => (node as YamlScalarNode)?.Value ?? "";

    private static string Join(string path, string key) => path.Length == 0 ? key : $"{path}.{key}";
}
