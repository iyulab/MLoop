namespace MLoop.Core.Models;

/// <summary>
/// The production model is not the file its promotion recorded and the project asked for that to be refused
/// (<c>integrity.refuse_changed_model</c> in mloop.yaml, <see cref="ModelIntegrity.Enforce"/>). The message names the file, both hashes and
/// the way back.
/// </summary>
public sealed class ChangedModelException(string message) : InvalidOperationException(message);
