#!/usr/bin/env python3
"""Line coverage per assembly from the cobertura files a test run left behind.

`dotnet test MLoop.slnx --coverage --coverage-output-format cobertura` writes one report per test
project, and each report only knows the lines its own project's tests reached. A line counts as
covered when ANY report hit it (union), which is the number a reader means by "the suite covers it".

    coverage-summary.py [--root DIR] [--floors FILE] [--top N] [--min-lines N] [--write-floors]
    coverage-summary.py --self-test

Exit status: 0 = every assembly at or above its floor, 1 = an assembly fell below its floor,
2 = no coverage report found (a run that measured nothing is a failure, not a pass).

The summary goes to stdout and, when GITHUB_STEP_SUMMARY is set, to the job summary as well.
Standard library only. Generated sources (obj/, bin/) and the test projects' own code are not counted.
"""
import argparse
import collections
import glob
import json
import math
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

EXCLUDED_SEGMENTS = ("obj", "bin", "tests")


def counted(filename):
    """A product source file: not generated, not a test project's own code."""
    parts = filename.replace("\\", "/").split("/")
    return not any(seg in EXCLUDED_SEGMENTS for seg in parts[:-1])


def find_reports(root):
    pattern = os.path.join(root, "**", "TestResults", "**", "*.cobertura.xml")
    found = set(glob.glob(pattern, recursive=True))
    # Reports written straight into a results directory named something else (--results-directory).
    found.update(glob.glob(os.path.join(root, "**", "*.cobertura.xml"), recursive=True))
    return sorted(found)


def merge(reports):
    """Union of line hits over reports -> {(assembly, filename, line): max hits}."""
    hits = {}
    for path in reports:
        tree = ET.parse(path)
        for package in tree.iter("package"):
            assembly = package.get("name")
            for cls in package.iter("class"):
                filename = cls.get("filename")
                if not filename or not counted(filename):
                    continue
                for line in cls.iter("line"):
                    key = (assembly, filename, int(line.get("number")))
                    count = int(line.get("hits"))
                    hits[key] = max(hits.get(key, 0), count)
    return hits


def tally(hits):
    by_assembly = collections.defaultdict(lambda: [0, 0])
    by_file = collections.defaultdict(lambda: [0, 0])
    for (assembly, filename, _), count in hits.items():
        by_assembly[assembly][1] += 1
        by_file[(assembly, filename)][1] += 1
        if count > 0:
            by_assembly[assembly][0] += 1
            by_file[(assembly, filename)][0] += 1
    return by_assembly, by_file


def percent(covered, total):
    return 100.0 * covered / total if total else 100.0


def short(filename):
    parts = filename.replace("\\", "/").split("/")
    for anchor in ("src", "tools"):
        if anchor in parts:
            return "/".join(parts[len(parts) - 1 - parts[::-1].index(anchor):])
    return "/".join(parts[-3:])


def load_floors(path):
    if not path or not os.path.exists(path):
        return {}, 0.0
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("floors", {}), float(data.get("tolerance", 0.0))


def render(by_assembly, by_file, floors, tolerance, top, min_lines, report_count):
    """Markdown summary and the list of assemblies below their floor."""
    below = []
    lines = ["## Line coverage (union of %d reports)" % report_count, ""]
    lines.append("| Assembly | Covered | Lines | % | Floor | |")
    lines.append("|---|---:|---:|---:|---:|---|")
    total = [0, 0]
    for assembly in sorted(by_assembly):
        covered, count = by_assembly[assembly]
        total[0] += covered
        total[1] += count
        pct = percent(covered, count)
        floor = floors.get(assembly)
        note = ""
        if floor is None:
            note = "no floor yet"
        elif pct + tolerance < floor:
            note = "**BELOW FLOOR**"
            below.append((assembly, pct, floor))
        elif pct - floor >= 5.0:
            note = "floor can rise to %.1f" % (math.floor(pct * 10) / 10 - 1.0)
        lines.append("| %s | %d | %d | %.1f | %s | %s |" % (
            assembly, covered, count, pct, "-" if floor is None else "%.1f" % floor, note))
    lines.append("| **all** | %d | %d | %.1f | | |" % (total[0], total[1], percent(*total)))
    stale = sorted(set(floors) - set(by_assembly))
    if stale:
        lines += ["", "Floors naming an assembly with no coverage data: %s" % ", ".join(stale)]
    rows = [(percent(c, t), c, t, key) for key, (c, t) in by_file.items() if t >= min_lines]
    rows.sort()
    lines += ["", "### Lowest-covered files (%d+ coverable lines)" % min_lines, "",
              "| % | Covered | Lines | File |", "|---:|---:|---:|---|"]
    for pct, covered, count, (_, filename) in rows[:top]:
        lines.append("| %.1f | %d | %d | `%s` |" % (pct, covered, count, short(filename)))
    return "\n".join(lines) + "\n", below


def write_floors(path, by_assembly, margin):
    floors = {a: max(0.0, math.floor(percent(c, t) * 10) / 10 - margin)
              for a, (c, t) in sorted(by_assembly.items())}
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({
            "basis": "line coverage, union of the test projects' reports; an assembly below its floor "
                     "fails the Linux CI leg. Raise a floor by editing this file, never lower it to pass.",
            "tolerance": 0.2,
            "floors": floors,
        }, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def self_test():
    """The tool must see what it claims to: union, exclusions, floors, and the empty case."""
    def report(directory, name, rows):
        classes = "".join(
            '<class name="c" filename="%s"><methods/><lines>%s</lines></class>' % (
                filename, "".join('<line number="%d" hits="%d" branch="False"/>' % (n, h) for n, h in lns))
            for filename, lns in rows)
        text = '<coverage><packages><package name="Asm">%s</package></packages></coverage>' % (
            "<classes>%s</classes>" % classes)
        with open(os.path.join(directory, name), "w", encoding="utf-8") as handle:
            handle.write(text)

    with tempfile.TemporaryDirectory() as tmp:
        results = os.path.join(tmp, "p1", "TestResults")
        os.makedirs(results)
        other = os.path.join(tmp, "p2", "TestResults")
        os.makedirs(other)
        report(results, "a.cobertura.xml", [
            ("/r/src/Asm/A.cs", [(1, 1), (2, 0), (3, 0)]),
            ("/r/src/Asm/obj/Release/G.g.cs", [(1, 0)]),       # generated: not counted
            ("/r/tests/Asm.Tests/T.cs", [(1, 1)]),             # test code: not counted
        ])
        report(other, "b.cobertura.xml", [("/r/src/Asm/A.cs", [(1, 0), (2, 5), (3, 0)])])
        reports = find_reports(tmp)
        assert len(reports) == 2, reports
        by_assembly, by_file = tally(merge(reports))
        assert by_assembly["Asm"] == [2, 3], by_assembly["Asm"]   # lines 1 and 2 hit by one report each
        assert len(by_file) == 1, by_file
        _, below = render(by_assembly, by_file, {"Asm": 90.0}, 0.0, 5, 1, 2)
        assert [b[0] for b in below] == ["Asm"], below            # 66.7 < 90 is seen
        _, below = render(by_assembly, by_file, {"Asm": 66.0}, 0.0, 5, 1, 2)
        assert not below, below                                   # at or above the floor is not
        assert not counted("/r/src/Asm/obj/x.cs") and not counted("C:\\r\\tests\\T.cs")
        assert counted("/r/src/Asm/Models/x.cs")
        assert find_reports(os.path.join(tmp, "p1", "TestResults", "nothing-here")) == []
    print("coverage-summary self-test: ok")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default=".")
    parser.add_argument("--floors", default=None)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--min-lines", type=int, default=40)
    parser.add_argument("--write-floors", action="store_true",
                        help="write --floors from this run's numbers minus --margin points")
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0

    reports = find_reports(args.root)
    if not reports:
        print("coverage-summary: no *.cobertura.xml under %s — the run measured nothing" % args.root,
              file=sys.stderr)
        return 2
    by_assembly, by_file = tally(merge(reports))
    if not by_assembly:
        print("coverage-summary: %d reports but no product source lines in them" % len(reports),
              file=sys.stderr)
        return 2
    if args.write_floors:
        if not args.floors:
            parser.error("--write-floors needs --floors FILE")
        write_floors(args.floors, by_assembly, args.margin)
        print("wrote %s" % args.floors)
    floors, tolerance = load_floors(args.floors)
    text, below = render(by_assembly, by_file, floors, tolerance, args.top, args.min_lines, len(reports))
    print(text)
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(text)
    for assembly, pct, floor in below:
        print("::error::%s line coverage %.1f%% is below its floor %.1f%%" % (assembly, pct, floor))
    return 1 if below else 0


if __name__ == "__main__":
    sys.exit(main())
