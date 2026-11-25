#!/usr/bin/env python3
"""The first thirty minutes of someone who installs the tool on Windows: the Windows counterpart of published-walk.sh.

A Windows x64 machine that has the .NET SDK, the generated tables and pictures standing in for "the user's own files", and the README and
guides of the version under test. A reader who knows only the README follows it on that machine. Nothing is published and nothing is
pushed; the machine gets one work folder and the tool, and `down` removes both.

    published-walk-windows.py up (--published <version> | --packed) --machine <name> [--ko-folder] [--fresh-profile]
    published-walk-windows.py exec --machine <name> [--ps] [--timeout <seconds>] -- <command ...>
    published-walk-windows.py fetch --machine <name> <path under the work folder> <local directory>
    published-walk-windows.py down --machine <name>

`up` generates the tables (walk-tables.py, with the Windows set) and the pictures (walk-pictures.py), takes the README and guides from
the tag v<version> - or, with --packed, packs the tool from this tree into a feed and takes them from the tree - copies them to
`%USERPROFILE%\\mloop-walk` and stops. The tool is NOT installed: the reader does that. What the generators know (the noise-free series,
the spike rows, the pictures' classes, boxes and defects) stays on this machine under the work directory and is never copied over: the
reader sees tables and pictures, whoever checks the answers compares them with the truth.

`--fresh-profile` gives the reader a machine with no mloop runtimes or encoders: the account's `%USERPROFILE%\\.mloop` (which cannot be
redirected for one process) is moved aside, and `down` removes the one the walk made and moves the account's own back. `up` refuses
when a moved-aside folder is already there - an earlier walk was not taken down.

The machine is reached by command execution only (walk_remote.py says how, and where its address and key are read from). `exec --ps`
runs the command in Windows PowerShell from the work folder, with `mloop` of the walk's own tool folder first on PATH, the walk's own
NuGet package cache, and DOTNET_ROOT set where the SDK is installed for the user only; it waits for the command however long it takes (up to --timeout). The command text is carried as a
script file, so Korean paths and quotes arrive as typed.

The reader's own folders, all under `%USERPROFILE%\\mloop-walk`: `data` (the tables), `docs` (the README and guides), `feed` (the packed
tool, with --packed), `work` (where a project is made), `with space`, with `--ko-folder` a folder with a Korean name that holds the CP949
spreadsheet tables, and a folder whose Korean name has a space in it, holding `pictures` and a `work` folder of its own for the picture
projects. The reader installs with `--tool-path <that folder>\\tools` so that a global tool the machine already has stays as it was;
`down` removes the folder and nothing else on the machine is changed.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WORK = os.path.join(os.path.expanduser("~"), ".cache", "mloop-published-walk-windows")
KO_FOLDER = "자료"
KO_RENAMES = {"regression-ko-train.csv": "주택가격-학습.csv", "regression-ko-new.csv": "주택가격-예측.csv"}
PICTURE_FOLDER = "사진 자료"  # a space and Korean letters, as a reader's own picture folder may well have
ROOT = "%USERPROFILE%\\mloop-walk"
sys.path.insert(0, HERE)
from walk_remote import download, execute, home, must, powershell, show, upload  # noqa: E402

SAVED_PROFILE = "mloop-walk-saved-profile"


def up(args):
    shutil.rmtree(WORK, ignore_errors=True)
    data, truth, docs, feed = (os.path.join(WORK, d) for d in ("data", "truth", "docs", "feed"))
    for d in (data, truth, docs, feed):
        os.makedirs(d)
    subprocess.run([sys.executable, "-I", os.path.join(HERE, "walk-tables.py"), data, truth, "windows"], check=True)
    pictures = os.path.join(WORK, "pictures")
    subprocess.run([sys.executable, "-I", os.path.join(HERE, "walk-pictures.py"), pictures, truth], check=True)
    for ko, ascii_name in ((v, k) for k, v in KO_RENAMES.items()):  # the archive carries ASCII names; the machine renames them
        os.replace(os.path.join(data, ko), os.path.join(data, ascii_name))

    if args.packed:
        # A version above anything published, so that an install from the feed takes the packed tool (as published-walk.sh).
        tree = next(line.split(">")[1].split("<")[0] for line in open(os.path.join(REPO, "Directory.Build.props"), encoding="utf-8")
                    if "<Version>" in line)
        major, minor = tree.split(".")[:2]
        version = "%s.%d.0.9999" % (major, int(minor) + 1)
        print("== pack %s" % version)
        subprocess.run(["dotnet", "pack", os.path.join(REPO, "tools", "MLoop.CLI"), "-c", "Release", "-p:Version=" + version, "-o", feed],
                       check=True, stdout=subprocess.DEVNULL)
        package = next(f for f in os.listdir(feed) if f.startswith("mloop.") and f.endswith(".nupkg"))
        with zipfile.ZipFile(os.path.join(feed, package)) as z, open(os.path.join(docs, "README.md"), "wb") as f:
            f.write(z.read("README.md"))
        for name in sorted(os.listdir(os.path.join(REPO, "docs"))):
            if name.endswith(".md"):
                shutil.copy(os.path.join(REPO, "docs", name), docs)
    else:
        version = args.published
        tag = "v" + version
        for name in ["README.md"] + [os.path.basename(f) for f in git(["ls-tree", "--name-only", tag, "docs/"]).split() if f.endswith(".md")]:
            path = "README.md" if name == "README.md" else "docs/" + name
            with open(os.path.join(docs, name), "wb") as f:
                f.write(git(["show", tag + ":" + path], binary=True))
    with open(os.path.join(WORK, "VERSION"), "w") as f:
        f.write(version + "\n")

    m = args.machine
    root = home(m) + "\\mloop-walk"
    must(m, 'rmdir /s /q "%s" 2>nul & mkdir "%s\\work" & mkdir "%s\\with space"' % (root, root, root), "making the work folder")
    with tempfile.TemporaryDirectory() as scratch:
        bundle = os.path.join(scratch, "walk.tar.gz")
        with tarfile.open(bundle, "w:gz") as tar:  # tar keeps the empty folders some of the wrong picture sets consist of
            for folder, name in ((data, "data"), (docs, "docs"), (feed, "feed"), (pictures, "pictures")):
                tar.add(folder, arcname=name)
        upload(m, bundle, root + "\\walk.tar.gz")
    # tar ships with Windows; Expand-Archive of Windows PowerShell 5.1 fails on some current builds
    must(m, 'tar -xzf "%s\\walk.tar.gz" -C "%s" && del /q "%s\\walk.tar.gz"' % (root, root, root), "unpacking", 300)
    place = "$r = Join-Path $env:USERPROFILE 'mloop-walk'; $p = Join-Path $r '%s'; New-Item -ItemType Directory (Join-Path $p 'work') -Force | Out-Null; " \
            "Move-Item (Join-Path $r 'pictures') (Join-Path $p 'pictures')" % PICTURE_FOLDER
    if args.ko_folder:
        moves = "; ".join("Move-Item (Join-Path $r 'data\\%s') (Join-Path $r '%s\\%s')" % (src, KO_FOLDER, dst) for src, dst in KO_RENAMES.items())
        place += "; New-Item -ItemType Directory (Join-Path $r '%s') | Out-Null; %s" % (KO_FOLDER, moves)
    must(m, powershell(place), "placing the tables and pictures")

    if args.fresh_profile:
        profile = home(m)
        if execute(m, 'if exist "%s\\%s" (exit 0) else (exit 1)' % (profile, SAVED_PROFILE), 60).get("exit_code") == 0:
            sys.exit("%s is already there: an earlier walk was not taken down; run `down` first" % SAVED_PROFILE)
        # The marker says `down` has a profile to give back, also when the account had no .mloop of its own.
        must(m, 'if exist "%s\\.mloop" (move "%s\\.mloop" "%s\\%s" >nul) else (mkdir "%s\\%s")'
             % (profile, profile, profile, SAVED_PROFILE, profile, SAVED_PROFILE), "moving the account's mloop folder aside")
    print("ready: %s holds data, docs of %s%s and '%s' (pictures, work); no mloop is installed there%s"
          % (root, version, ", feed" if args.packed else "", PICTURE_FOLDER,
             "; the account's .mloop is moved aside" if args.fresh_profile else ""))


def git(args, binary=False):
    out = subprocess.run(["git", "-C", REPO] + args, check=True, capture_output=True).stdout
    return out if binary else out.decode("utf-8")


# What every `exec --ps` command starts with: the walk's own tool first on PATH (a global tool the machine has stays as it is),
# its own package cache (a packed version is built again each round under one number), the SDK found where it was installed
# for the user only, and UTF-8 both ways.
PREAMBLE = """$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
$OutputEncoding = [Console]::OutputEncoding
$walk = Join-Path $env:USERPROFILE 'mloop-walk'
if (-not $env:DOTNET_ROOT -and -not (Test-Path (Join-Path $env:ProgramFiles 'dotnet\\dotnet.exe'))) { $env:DOTNET_ROOT = Join-Path $env:LOCALAPPDATA 'Microsoft\\dotnet' }
$env:NUGET_PACKAGES = Join-Path $walk 'nuget'
$env:PATH = (Join-Path $walk 'tools') + ';' + $env:PATH
Set-Location $walk
"""


def run_script(machine, text, timeout):
    """`text` as a PowerShell script on the machine, waited for however long it takes (up to `timeout`); its output and exit code."""
    root = home(machine) + "\\mloop-walk"
    job = root + "\\jobs\\" + uuid.uuid4().hex[:10]
    with tempfile.TemporaryDirectory() as scratch:
        local = os.path.join(scratch, "job.ps1")
        with open(local, "w", encoding="utf-8-sig", newline="\r\n") as f:  # the mark makes Windows PowerShell read it as UTF-8
            f.write(PREAMBLE + text + "\nexit $LASTEXITCODE\n")
        must(machine, 'if not exist "%s\\jobs" mkdir "%s\\jobs"' % (root, root), "making the job folder")
        upload(machine, local, job + ".ps1")
    command = 'powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%s.ps1" > "%s.log" 2>&1 & echo !errorlevel! > "%s.exit"' % (job, job, job)
    must(machine, powershell("Start-Process cmd -WindowStyle Hidden -ArgumentList '/v:on','/c','%s'" % command.replace("'", "''")), "starting the command")
    deadline, wait, code = time.time() + timeout, 1.0, ""
    while time.time() < deadline:
        time.sleep(wait)
        wait = min(wait * 1.5, 15.0)
        code = execute(machine, 'type "%s.exit" 2>nul' % job, 60, retries=5).get("stdout", "").strip()
        if code:
            break
    read = "[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false; if (Test-Path '%s.log') { Get-Content -Encoding UTF8 -Tail 600 '%s.log' }" % (job, job)
    sys.stdout.write(execute(machine, powershell(read), 120, retries=5).get("stdout", ""))
    if not code:
        sys.stderr.write("[still running after %d s; its output so far is above, the rest goes to %s.log]\n" % (timeout, job))
        return 124
    return int(code)


def fetch(args):
    m = args.machine
    root = home(m) + "\\mloop-walk"
    bundle_name = "fetch-%s.tar.gz" % uuid.uuid4().hex[:8]
    pack = "$r = Join-Path $env:USERPROFILE 'mloop-walk'; $s = Get-Item -LiteralPath (Join-Path $r '%s'); " \
           "tar -czf (Join-Path $r '%s') -C $s.PSParentPath.Replace('Microsoft.PowerShell.Core\\FileSystem::','') $s.Name" % (args.source.replace("'", "''"), bundle_name)
    must(m, powershell(pack), "packing " + args.source, 300)
    os.makedirs(args.target, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        local = os.path.join(scratch, bundle_name)
        download(m, root + "\\" + bundle_name, local)
        with tarfile.open(local) as tar:
            tar.extractall(args.target, filter="data")
    execute(m, 'del /q "%s\\%s"' % (root, bundle_name), 60)
    print("fetched %s into %s" % (args.source, args.target))


def down(args):
    m = args.machine
    profile = home(m)
    # Only the processes started from the walk's folder are stopped.
    stop = ("powershell -NoProfile -Command \"Get-Process | Where-Object { $_.Path -like '*\\mloop-walk\\*' } | Stop-Process -Force\"")
    show(execute(m, "%s & cd /d %%USERPROFILE%% & rmdir /s /q mloop-walk & echo removed the walk folder" % stop, 180))
    if execute(m, 'if exist "%s\\%s" (exit 0) else (exit 1)' % (profile, SAVED_PROFILE), 60).get("exit_code") == 0:
        must(m, 'if exist "%s\\.mloop" rmdir /s /q "%s\\.mloop"' % (profile, profile), "removing the walk's mloop folder", 180)
        # An empty saved folder stands for «the account had none».
        must(m, 'dir /b /a "%s\\%s" | findstr . >nul && (move "%s\\%s" "%s\\.mloop" >nul) || (rmdir "%s\\%s")'
             % (profile, SAVED_PROFILE, profile, SAVED_PROFILE, profile, profile, SAVED_PROFILE), "giving the account its mloop folder back")
        print("the account's .mloop is back as it was")
    shutil.rmtree(WORK, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("up")
    build = u.add_mutually_exclusive_group(required=True)
    build.add_argument("--published")
    build.add_argument("--packed", action="store_true", help="pack the tool from this tree into a feed on the machine")
    u.add_argument("--machine", required=True)
    u.add_argument("--ko-folder", action="store_true", help="also put the CP949 spreadsheet tables in a folder with a Korean name")
    u.add_argument("--fresh-profile", action="store_true", help="move the account's .mloop (runtimes, encoders) aside until `down`")
    f = sub.add_parser("fetch")
    f.add_argument("--machine", required=True)
    f.add_argument("source", help="a file or folder under the work folder on the machine")
    f.add_argument("target")
    e = sub.add_parser("exec")
    e.add_argument("--machine", required=True)
    e.add_argument("--ps", action="store_true")
    e.add_argument("--timeout", type=int, default=120)
    e.add_argument("command", nargs=argparse.REMAINDER)
    d = sub.add_parser("down")
    d.add_argument("--machine", required=True)
    args = p.parse_args()
    if args.cmd == "up":
        up(args)
    elif args.cmd == "down":
        down(args)
    elif args.cmd == "fetch":
        fetch(args)
    else:
        words = args.command[1:] if args.command[:1] == ["--"] else args.command
        text = " ".join(words)
        sys.exit(run_script(args.machine, text, args.timeout) if args.ps else show(execute(args.machine, text, args.timeout)))


if __name__ == "__main__":
    main()
