using Xunit;

// ISSUE-mloop-20260905-json-test-console-capture-race: System.Console.Out and
// Spectre.Console.AnsiConsole.Console are process-wide statics. `[Collection("FileSystem")]`
// only guarantees no two tests *within* that collection run concurrently — it does nothing to
// stop an unrelated test in a different (default) collection from calling into production code
// that writes to AnsiConsole/Console mid-await, corrupting whichever StringWriter a
// FileSystem-collection test currently has the statics pointed at. Disabling assembly-wide
// parallelization is the only fix that actually closes the race (confirmed: three consecutive
// full-suite runs each failed a different assertion in a different --json capture test, while
// the same tests run 100% clean in isolation or together).
[assembly: CollectionBehavior(DisableTestParallelization = true)]
