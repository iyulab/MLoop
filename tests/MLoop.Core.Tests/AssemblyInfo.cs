using Xunit;

// This assembly mutates process-wide statics: System.Console.Out (output capture in
// CsvDataLoaderTests) and MLoop.Core.AutoML.DeepLearningRegistry (registration in the
// DeepLearning tests, read by the loader-factory tests).
//
// `[Collection(...)]` cannot make that safe. It serializes the classes that share the name and
// says nothing about the rest of the assembly — so an unrelated class calling into production
// code that writes to Console, or that resolves a loader through the registry, still runs
// concurrently with the class that has the static pointed somewhere else. Only assembly-wide
// serialization closes it, which is the same conclusion MLoop.Tests reached, on the same statics,
// with three consecutive full-suite runs each failing a different assertion.
//
// Measured cost of this line: 5m03s -> 7m40s on 1167 tests. The alternative — removing the
// mutations — means production code no longer writing to Console from a library, which is a
// change to the product rather than to its tests.
[assembly: CollectionBehavior(DisableTestParallelization = true)]
