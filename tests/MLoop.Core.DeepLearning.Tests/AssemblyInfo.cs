using Xunit;

// ObjectDetectionDataLoaderTests registers a module into the process-wide
// MLoop.Core.AutoML.DeepLearningRegistry, which every loader-factory resolution in this assembly
// reads. Serialized for the same reason as MLoop.Tests and MLoop.Core.Tests — a collection only
// orders the classes that share it, not the assembly around them.
[assembly: CollectionBehavior(DisableTestParallelization = true)]
