using MLoop.CLI.Infrastructure.Configuration;
using MLoop.CLI.Infrastructure.FileSystem;

namespace MLoop.Tests.Infrastructure.FileSystem;

[Collection("FileSystem")]
public class ModelNameResolverTests : IDisposable
{
    private readonly string _testProjectRoot;
    private readonly string _originalDirectory;
    private readonly IFileSystemManager _fileSystem;
    private readonly IProjectDiscovery _projectDiscovery;
    private readonly ConfigLoader _configLoader;
    private readonly ModelNameResolver _resolver;

    public ModelNameResolverTests()
    {
        _originalDirectory = Directory.GetCurrentDirectory();

        _testProjectRoot = Path.Combine(Path.GetTempPath(), "mloop-mnr-test-" + Guid.NewGuid());
        Directory.CreateDirectory(_testProjectRoot);
        Directory.CreateDirectory(Path.Combine(_testProjectRoot, ".mloop"));

        _fileSystem = new FileSystemManager();
        _projectDiscovery = new ProjectDiscovery(_fileSystem);

        Directory.SetCurrentDirectory(_testProjectRoot);

        _configLoader = new ConfigLoader(_fileSystem, _projectDiscovery);
        _resolver = new ModelNameResolver(_fileSystem, _projectDiscovery, _configLoader);
    }

    public void Dispose()
    {
        try { Directory.SetCurrentDirectory(_originalDirectory); }
        catch { try { Directory.SetCurrentDirectory(Path.GetTempPath()); } catch { } }

        if (Directory.Exists(_testProjectRoot))
        {
            try { Directory.Delete(_testProjectRoot, recursive: true); } catch { }
        }
    }

    // --- Resolve tests ---

    [Fact]
    public void Resolve_Null_ReturnsDefault()
    {
        Assert.Equal("default", _resolver.Resolve(null));
    }

    [Fact]
    public void Resolve_Empty_ReturnsDefault()
    {
        Assert.Equal("default", _resolver.Resolve(""));
    }

    [Fact]
    public void Resolve_Whitespace_ReturnsDefault()
    {
        Assert.Equal("default", _resolver.Resolve("   "));
    }

    [Fact]
    public void Resolve_MixedCase_ReturnsLowercase()
    {
        Assert.Equal("mymodel", _resolver.Resolve("MyModel"));
    }

    [Fact]
    public void Resolve_WithSpaces_TrimsAndLowercases()
    {
        Assert.Equal("mymodel", _resolver.Resolve("  MyModel  "));
    }

    [Fact]
    public void Resolve_AlreadyLowercase_ReturnsSame()
    {
        Assert.Equal("my-model", _resolver.Resolve("my-model"));
    }

    // --- IsValidName tests ---

    [Theory]
    [InlineData("default", true)]
    [InlineData("my-model", true)]
    [InlineData("ab", true)]           // minimum length
    [InlineData("a1", true)]
    [InlineData("model-v2", true)]
    [InlineData("churn-predictor", true)]
    public void IsValidName_ValidNames(string name, bool expected)
    {
        Assert.Equal(expected, _resolver.IsValidName(name));
    }

    [Theory]
    [InlineData("", false)]            // empty
    [InlineData("a", false)]           // too short
    [InlineData("-model", false)]       // starts with hyphen
    [InlineData("model-", false)]       // ends with hyphen
    [InlineData("My-Model", false)]     // uppercase
    [InlineData("model name", false)]   // space
    [InlineData("model_name", false)]   // underscore
    [InlineData("model.name", false)]   // dot
    [InlineData("1model", false)]       // starts with digit
    public void IsValidName_InvalidNames(string name, bool expected)
    {
        Assert.Equal(expected, _resolver.IsValidName(name));
    }

    [Theory]
    [InlineData("staging")]
    [InlineData("production")]
    [InlineData("temp")]
    [InlineData("cache")]
    [InlineData("index")]
    [InlineData("registry")]
    public void IsValidName_ReservedNames_ReturnsFalse(string name)
    {
        Assert.False(_resolver.IsValidName(name));
    }

    [Fact]
    public void IsValidName_ReservedNameCaseInsensitive_ReturnsFalse()
    {
        // Reserved names use OrdinalIgnoreCase, but IsValidName checks regex
        // which requires lowercase — so "STAGING" fails at regex, not reserved check
        Assert.False(_resolver.IsValidName("STAGING"));
    }

    [Fact]
    public void IsValidName_MaxLength50_ReturnsTrue()
    {
        var name = "a" + new string('b', 49); // 50 chars
        Assert.True(_resolver.IsValidName(name));
    }

    [Fact]
    public void IsValidName_Over50_ReturnsFalse()
    {
        var name = "a" + new string('b', 50); // 51 chars
        Assert.False(_resolver.IsValidName(name));
    }

    // --- GetModelPath tests ---

    [Fact]
    public void GetModelPath_ReturnsCorrectPath()
    {
        var path = _resolver.GetModelPath("my-model");
        Assert.EndsWith(Path.Combine("models", "my-model"), path);
    }

    [Fact]
    public void GetModelPath_ResolvesCase()
    {
        var path = _resolver.GetModelPath("MyModel");
        Assert.EndsWith(Path.Combine("models", "mymodel"), path);
    }

    // --- Exists tests ---

    [Fact]
    public void Exists_NoDirectory_ReturnsFalse()
    {
        Assert.False(_resolver.Exists("nonexistent"));
    }

    [Fact]
    public void Exists_DirectoryExists_ReturnsTrue()
    {
        var modelPath = _resolver.GetModelPath("test-model");
        Directory.CreateDirectory(modelPath);

        Assert.True(_resolver.Exists("test-model"));
    }

    // --- CreateAsync tests ---

    [Fact]
    public async Task CreateAsync_ValidName_CreatesDirectoryStructure()
    {
        var definition = new ModelDefinition
        {
            Task = "binary-classification",
            Label = "target"
        };

        await _resolver.CreateAsync("my-model", definition);

        var modelPath = _resolver.GetModelPath("my-model");
        Assert.True(Directory.Exists(modelPath));
        Assert.True(Directory.Exists(Path.Combine(modelPath, "staging")));
        Assert.True(Directory.Exists(Path.Combine(modelPath, "production")));
    }

    [Fact]
    public async Task CreateAsync_InvalidName_ThrowsArgumentException()
    {
        var definition = new ModelDefinition { Task = "regression", Label = "price" };

        await Assert.ThrowsAsync<ArgumentException>(
            () => _resolver.CreateAsync("1invalid", definition));
    }

    [Fact]
    public async Task CreateAsync_DuplicateName_ThrowsInvalidOperationException()
    {
        var definition = new ModelDefinition { Task = "regression", Label = "price" };
        await _resolver.CreateAsync("dup-model", definition);

        await Assert.ThrowsAsync<InvalidOperationException>(
            () => _resolver.CreateAsync("dup-model", definition));
    }

    // --- RemoveAsync tests ---

    [Fact]
    public async Task RemoveAsync_ExistingModel_RemovesDirectory()
    {
        var definition = new ModelDefinition { Task = "regression", Label = "price" };
        await _resolver.CreateAsync("removable", definition);
        Assert.True(_resolver.Exists("removable"));

        await _resolver.RemoveAsync("removable");

        Assert.False(_resolver.Exists("removable"));
    }

    [Fact]
    public async Task RemoveAsync_DefaultModel_ThrowsInvalidOperationException()
    {
        await Assert.ThrowsAsync<InvalidOperationException>(
            () => _resolver.RemoveAsync("default"));
    }

    [Fact]
    public async Task RemoveAsync_NonexistentModel_ThrowsFileNotFoundException()
    {
        await Assert.ThrowsAsync<FileNotFoundException>(
            () => _resolver.RemoveAsync("ghost"));
    }

    // --- EnsureModelDirectoryAsync tests ---

    [Fact]
    public async Task EnsureModelDirectoryAsync_CreatesSubdirectories()
    {
        await _resolver.EnsureModelDirectoryAsync("new-model");

        var modelPath = _resolver.GetModelPath("new-model");
        Assert.True(Directory.Exists(Path.Combine(modelPath, "staging")));
        Assert.True(Directory.Exists(Path.Combine(modelPath, "production")));
    }

    // --- Constructor tests ---

    [Fact]
    public void Constructor_NullFileSystem_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(
            () => new ModelNameResolver(null!, _projectDiscovery, _configLoader));
    }

    [Fact]
    public void Constructor_NullProjectDiscovery_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(
            () => new ModelNameResolver(_fileSystem, null!, _configLoader));
    }

    [Fact]
    public void Constructor_NullConfigLoader_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(
            () => new ModelNameResolver(_fileSystem, _projectDiscovery, null!));
    }
}
