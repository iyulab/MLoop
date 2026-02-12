using System.Text.Json;
using MLoop.CLI.Commands;
using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;

namespace MLoop.Tests.Commands;

public class TrainCommandSyncYamlTests
{
    #region Fakes

    private class FakeFileSystemManager : IFileSystemManager
    {
        private readonly Dictionary<string, string> _textFiles = new();
        private readonly Dictionary<string, string> _jsonFiles = new();

        public void SetTextFile(string path, string content) => _textFiles[path] = content;

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
            _jsonFiles[path] = JsonSerializer.Serialize(content);
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
        public string FindRoot() => "/project";
        public string FindRoot(string startingDirectory) => "/project";
        public bool IsProjectRoot(string path) => path == "/project";
        public void EnsureProjectRoot() { }
        public string GetMLoopDirectory(string projectRoot) => "/project/.mloop";
    }

    #endregion

    private readonly FakeFileSystemManager _fs = new();
    private readonly FakeProjectDiscovery _discovery = new();
    private readonly ConfigLoader _configLoader;

    public TrainCommandSyncYamlTests()
    {
        _configLoader = new ConfigLoader(_fs, _discovery);
    }

    [Fact]
    public async Task SyncYamlConfig_NoCLIOverride_DoesNotWrite()
    {
        var userConfig = new MLoopConfig
        {
            Project = "test",
            Models = new Dictionary<string, ModelDefinition>
            {
                ["default"] = new() { Task = "regression", Label = "Price" }
            }
        };
        var effective = new ModelDefinition { Task = "regression", Label = "Price" };

        await TrainCommand.SyncYamlConfigAsync(_configLoader, userConfig, "default", effective, cliLabel: null, cliTask: null);

        Assert.Null(_fs.LastWrittenTextPath);
    }

    [Fact]
    public async Task SyncYamlConfig_CLILabelDiffers_UpdatesYaml()
    {
        var userConfig = new MLoopConfig
        {
            Project = "test",
            Models = new Dictionary<string, ModelDefinition>
            {
                ["default"] = new() { Task = "regression", Label = "OldLabel" }
            }
        };
        var effective = new ModelDefinition { Task = "regression", Label = "NewLabel" };

        await TrainCommand.SyncYamlConfigAsync(_configLoader, userConfig, "default", effective, cliLabel: "NewLabel", cliTask: null);

        Assert.NotNull(_fs.LastWrittenTextContent);
        Assert.Contains("NewLabel", _fs.LastWrittenTextContent);
        Assert.Equal("NewLabel", userConfig.Models["default"].Label);
    }

    [Fact]
    public async Task SyncYamlConfig_CLITaskDiffers_UpdatesYaml()
    {
        var userConfig = new MLoopConfig
        {
            Project = "test",
            Models = new Dictionary<string, ModelDefinition>
            {
                ["default"] = new() { Task = "regression", Label = "Price" }
            }
        };
        var effective = new ModelDefinition { Task = "binary-classification", Label = "Price" };

        await TrainCommand.SyncYamlConfigAsync(_configLoader, userConfig, "default", effective, cliLabel: null, cliTask: "binary-classification");

        Assert.NotNull(_fs.LastWrittenTextContent);
        Assert.Contains("binary-classification", _fs.LastWrittenTextContent);
        Assert.Equal("binary-classification", userConfig.Models["default"].Task);
    }

    [Fact]
    public async Task SyncYamlConfig_CLILabelSameAsYaml_DoesNotWrite()
    {
        var userConfig = new MLoopConfig
        {
            Project = "test",
            Models = new Dictionary<string, ModelDefinition>
            {
                ["default"] = new() { Task = "regression", Label = "Price" }
            }
        };
        var effective = new ModelDefinition { Task = "regression", Label = "Price" };

        await TrainCommand.SyncYamlConfigAsync(_configLoader, userConfig, "default", effective, cliLabel: "Price", cliTask: null);

        Assert.Null(_fs.LastWrittenTextPath);
    }

    [Fact]
    public async Task SyncYamlConfig_ModelNotInYaml_CreatesNewEntry()
    {
        var userConfig = new MLoopConfig
        {
            Project = "test",
            Models = new Dictionary<string, ModelDefinition>()
        };
        var effective = new ModelDefinition { Task = "regression", Label = "Price" };

        await TrainCommand.SyncYamlConfigAsync(_configLoader, userConfig, "custom-model", effective, cliLabel: "Price", cliTask: null);

        Assert.NotNull(_fs.LastWrittenTextContent);
        Assert.True(userConfig.Models.ContainsKey("custom-model"));
        Assert.Equal("Price", userConfig.Models["custom-model"].Label);
        Assert.Equal("regression", userConfig.Models["custom-model"].Task);
    }

    [Fact]
    public async Task SyncYamlConfig_BothLabelAndTaskOverridden_UpdatesBoth()
    {
        var userConfig = new MLoopConfig
        {
            Project = "test",
            Models = new Dictionary<string, ModelDefinition>
            {
                ["default"] = new() { Task = "regression", Label = "OldLabel" }
            }
        };
        var effective = new ModelDefinition { Task = "binary-classification", Label = "NewLabel" };

        await TrainCommand.SyncYamlConfigAsync(_configLoader, userConfig, "default", effective, cliLabel: "NewLabel", cliTask: "binary-classification");

        Assert.Equal("NewLabel", userConfig.Models["default"].Label);
        Assert.Equal("binary-classification", userConfig.Models["default"].Task);
    }
}
