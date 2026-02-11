using System.Text.Json;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;

namespace MLoop.Tests.Configuration;

public class ConfigLoaderTests
{
    #region Fakes

    private class FakeFileSystemManager : IFileSystemManager
    {
        private readonly Dictionary<string, string> _textFiles = new();
        private readonly Dictionary<string, string> _jsonFiles = new();

        public void SetTextFile(string path, string content) => _textFiles[path] = content;
        public void SetJsonFile<T>(string path, T content) =>
            _jsonFiles[path] = JsonSerializer.Serialize(content);

        public string? LastWrittenJsonPath { get; private set; }
        public string? LastWrittenJsonContent { get; private set; }
        public string? LastWrittenTextPath { get; private set; }
        public string? LastWrittenTextContent { get; private set; }

        public bool FileExists(string path) => _textFiles.ContainsKey(path) || _jsonFiles.ContainsKey(path);
        public bool DirectoryExists(string path) => true;
        public string CombinePath(params string[] paths) => string.Join("/", paths);
        public string GetAbsolutePath(string path) => path;

        public Task<T> ReadJsonAsync<T>(string path, CancellationToken cancellationToken = default)
        {
            if (!_jsonFiles.TryGetValue(path, out var json))
                throw new FileNotFoundException(path);
            return Task.FromResult(JsonSerializer.Deserialize<T>(json)!);
        }

        public Task<string> ReadTextAsync(string path, CancellationToken cancellationToken = default)
        {
            if (!_textFiles.TryGetValue(path, out var text))
                throw new FileNotFoundException(path);
            return Task.FromResult(text);
        }

        public Task WriteJsonAsync<T>(string path, T content, CancellationToken cancellationToken = default)
        {
            LastWrittenJsonPath = path;
            LastWrittenJsonContent = JsonSerializer.Serialize(content);
            _jsonFiles[path] = LastWrittenJsonContent;
            return Task.CompletedTask;
        }

        public Task WriteTextAsync(string path, string content, CancellationToken cancellationToken = default)
        {
            LastWrittenTextPath = path;
            LastWrittenTextContent = content;
            _textFiles[path] = content;
            return Task.CompletedTask;
        }

        public Task CreateDirectoryAsync(string path, CancellationToken cancellationToken = default) => Task.CompletedTask;
        public Task CopyFileAsync(string src, string dst, bool overwrite = false, CancellationToken cancellationToken = default) => Task.CompletedTask;
        public Task DeleteDirectoryAsync(string path, CancellationToken cancellationToken = default) => Task.CompletedTask;
        public IEnumerable<string> GetFiles(string path, string searchPattern = "*", bool recursive = false) => [];
    }

    private class FakeProjectDiscovery : IProjectDiscovery
    {
        public string ProjectRoot { get; set; } = "/project";
        public string MLoopDir { get; set; } = "/project/.mloop";

        public string FindRoot() => ProjectRoot;
        public string FindRoot(string startingDirectory) => ProjectRoot;
        public bool IsProjectRoot(string path) => path == ProjectRoot;
        public void EnsureProjectRoot() { }
        public string GetMLoopDirectory(string projectRoot) => MLoopDir;
    }

    #endregion

    private readonly FakeFileSystemManager _fs = new();
    private readonly FakeProjectDiscovery _discovery = new();
    private readonly ConfigLoader _loader;

    public ConfigLoaderTests()
    {
        _loader = new ConfigLoader(_fs, _discovery);
    }

    #region Constructor Tests

    [Fact]
    public void Constructor_NullFileSystem_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new ConfigLoader(null!, _discovery));
    }

    [Fact]
    public void Constructor_NullProjectDiscovery_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new ConfigLoader(_fs, null!));
    }

    #endregion

    #region LoadProjectConfigAsync Tests

    [Fact]
    public async Task LoadProjectConfigAsync_NoConfigFile_ReturnsDefault()
    {
        var result = await _loader.LoadProjectConfigAsync();

        Assert.NotNull(result);
        Assert.Null(result.Project);
        Assert.Empty(result.Models);
    }

    [Fact]
    public async Task LoadProjectConfigAsync_ConfigExists_ReturnsConfig()
    {
        var config = new MLoopConfig
        {
            Project = "test-project",
            Models = new Dictionary<string, ModelDefinition>
            {
                ["default"] = new() { Task = "regression", Label = "Price" }
            }
        };
        _fs.SetJsonFile("/project/.mloop/config.json", config);

        var result = await _loader.LoadProjectConfigAsync();

        Assert.Equal("test-project", result.Project);
        Assert.Single(result.Models);
        Assert.Equal("regression", result.Models["default"].Task);
    }

    #endregion

    #region LoadUserConfigAsync Tests

    [Fact]
    public async Task LoadUserConfigAsync_NoYamlFile_ReturnsDefault()
    {
        var result = await _loader.LoadUserConfigAsync();

        Assert.NotNull(result);
        Assert.Null(result.Project);
        Assert.Empty(result.Models);
    }

    [Fact]
    public async Task LoadUserConfigAsync_NewFormat_ParsesCorrectly()
    {
        var yaml = @"project: my-ml-project
models:
  default:
    task: regression
    label: Price
    training:
      time_limit_seconds: 600
      metric: r2
      test_split: 0.3
";
        _fs.SetTextFile("/project/mloop.yaml", yaml);

        var result = await _loader.LoadUserConfigAsync();

        Assert.Equal("my-ml-project", result.Project);
        Assert.True(result.Models.ContainsKey("default"));
        Assert.Equal("regression", result.Models["default"].Task);
        Assert.Equal("Price", result.Models["default"].Label);
    }

    [Fact]
    public async Task LoadUserConfigAsync_OldFormat_ConvertsToNewFormat()
    {
        var yaml = @"project_name: legacy-project
task: binary-classification
label_column: Target
time_limit_seconds: 120
metric: accuracy
test_split: 0.15
";
        _fs.SetTextFile("/project/mloop.yaml", yaml);

        var result = await _loader.LoadUserConfigAsync();

        Assert.Equal("legacy-project", result.Project);
        Assert.True(result.Models.ContainsKey("default"));
        Assert.Equal("binary-classification", result.Models["default"].Task);
        Assert.Equal("Target", result.Models["default"].Label);
        Assert.Equal(120, result.Models["default"].Training?.TimeLimitSeconds);
        Assert.Equal("accuracy", result.Models["default"].Training?.Metric);
        Assert.Equal(0.15, result.Models["default"].Training?.TestSplit);
    }

    [Fact]
    public async Task LoadUserConfigAsync_OldFormatNoTaskLabel_NoDefaultModel()
    {
        var yaml = @"project_name: partial-project
";
        _fs.SetTextFile("/project/mloop.yaml", yaml);

        var result = await _loader.LoadUserConfigAsync();

        Assert.Equal("partial-project", result.Project);
        Assert.Empty(result.Models);
    }

    [Fact]
    public async Task LoadUserConfigAsync_EmptyYaml_ReturnsDefault()
    {
        _fs.SetTextFile("/project/mloop.yaml", "");

        var result = await _loader.LoadUserConfigAsync();

        Assert.NotNull(result);
    }

    [Fact]
    public async Task LoadUserConfigAsync_MultiModel_ParsesAll()
    {
        var yaml = @"project: multi-model
models:
  price-model:
    task: regression
    label: Price
  category-model:
    task: multiclass-classification
    label: Category
";
        _fs.SetTextFile("/project/mloop.yaml", yaml);

        var result = await _loader.LoadUserConfigAsync();

        Assert.Equal("multi-model", result.Project);
        Assert.Equal(2, result.Models.Count);
        Assert.Equal("regression", result.Models["price-model"].Task);
        Assert.Equal("multiclass-classification", result.Models["category-model"].Task);
    }

    #endregion

    #region SaveProjectConfigAsync Tests

    [Fact]
    public async Task SaveProjectConfigAsync_WritesJson()
    {
        var config = new MLoopConfig { Project = "save-test" };

        await _loader.SaveProjectConfigAsync(config);

        Assert.Equal("/project/.mloop/config.json", _fs.LastWrittenJsonPath);
        Assert.Contains("save-test", _fs.LastWrittenJsonContent);
    }

    #endregion

    #region SaveUserConfigAsync Tests

    [Fact]
    public async Task SaveUserConfigAsync_WritesYaml()
    {
        var config = new MLoopConfig { Project = "yaml-save" };

        await _loader.SaveUserConfigAsync(config);

        Assert.Equal("/project/mloop.yaml", _fs.LastWrittenTextPath);
        Assert.NotNull(_fs.LastWrittenTextContent);
        Assert.Contains("yaml-save", _fs.LastWrittenTextContent);
    }

    #endregion
}
