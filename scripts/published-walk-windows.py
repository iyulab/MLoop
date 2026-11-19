#!/usr/bin/env python3
"""The first thirty minutes of someone who installs the tool on Windows: the Windows counterpart of published-walk.sh.

A clean Windows x64 machine that has only the .NET SDK, the generated tables standing in for "the user's own file" and the README and
guides of the version under test. A reader who knows only the README follows it on that machine. Nothing is published and nothing is
pushed; the machine gets one work folder and the tool, and `down` removes both.

    published-walk-windows.py up --published <version> --machine <name> [--ko-folder]
    published-walk-windows.py exec --machine <name> [--ps] [--timeout <seconds>] -- <command ...>
    published-walk-windows.py down --machine <name>

`up` generates the tables (walk-tables.py, with the Windows set), takes the README and guides from the tag v<version>, copies them to
`%USERPROFILE%\\mloop-walk\\{data,docs}` and stops. The tool is NOT installed: the reader does that (`dotnet tool install --global mloop`,
or the release binary). The truth (the noise-free series, the spike rows, ...) stays on this machine under the work directory and is
never copied over: the reader sees tables, whoever checks the answers compares them with the truth.

The machine is reached by command execution only: a remote endpoint that takes `POST <url>/api/v1/execute` with a JSON body
`{"command", "timeout_secs"}` and a bearer key, running the command in cmd.exe. Its address and key are read from the file named by
MLOOP_WALK_WINDOWS_ENV (lines `<name>_url=...` and `<name>_key=...`); neither is printed or stored anywhere else. `exec --ps` runs the
command in Windows PowerShell with the command text carried as an encoded script, so Korean paths and quotes arrive as typed.

The reader's own folders: `data` (the tables), `docs` (the README and guides), `work` (where a project is made), and with `--ko-folder` a
folder with a Korean name that holds the CP949 spreadsheet tables, and `with space` (a folder whose name has a space, for the standalone
binary). Everything a round creates sits under `%USERPROFILE%\\mloop-walk`; `down` removes that folder and, when the round installed it,
the global tool (`dotnet tool uninstall --global mloop`; a machine that had the tool before `up` keeps it, and the reader then installs into
a `--tool-path` of its own under that folder and runs it by its path), and nothing else on the machine is changed.
"""
import argparse
import base64
import io
import json
import os
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WORK = os.path.join(os.path.expanduser("~"), ".cache", "mloop-published-walk-windows")
KO_FOLDER = "자료"
KO_RENAMES = {"regression-ko-train.csv": "주택가격-학습.csv", "regression-ko-new.csv": "주택가격-예측.csv"}
ROOT = "%USERPROFILE%\\mloop-walk"


def endpoint(machine):
    path = os.environ.get("MLOOP_WALK_WINDOWS_ENV")
    if not path:
        sys.exit("MLOOP_WALK_WINDOWS_ENV names the file with <name>_url and <name>_key lines; it is not set")
    values = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip()
    try:
        return values[machine + "_url"].rstrip("/"), values[machine + "_key"]
    except KeyError:
        sys.exit("no %s_url / %s_key in the endpoint file" % (machine, machine))


def execute(machine, command, timeout=120):
    url, key = endpoint(machine)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False  # the endpoint serves a self-signed certificate
    ctx.verify_mode = ssl.CERT_NONE
    body = json.dumps({"command": command, "timeout_secs": min(timeout, 300)}).encode("utf-8")
    request = urllib.request.Request(url + "/api/v1/execute", data=body,
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with urllib.request.urlopen(request, context=ctx, timeout=timeout + 30) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


def powershell(script):
    """A cmd.exe command line that runs `script` in Windows PowerShell; the text travels as an encoded (UTF-16) script."""
    script = "$ProgressPreference = 'SilentlyContinue'; $ErrorActionPreference = 'Stop'; " + script
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return "powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand " + encoded


def show(result):
    sys.stdout.write(result.get("stdout") or "")
    if result.get("stderr"):
        sys.stderr.write(result["stderr"])
    if result.get("timed_out"):
        sys.stderr.write("[timed out]\n")
    sys.stdout.flush()
    return 0 if result.get("exit_code") == 0 else (result.get("exit_code") or 1)


# cmd.exe skips the rest of a line when an `if` in it is false: never chain a command after an `if`.
def must(machine, command, what, timeout=120):
    result = execute(machine, command, timeout)
    if result.get("exit_code") != 0:
        sys.exit("%s failed (exit %s): %s" % (what, result.get("exit_code"), (result.get("output") or "")[:600]))
    return result


def up(args):
    shutil.rmtree(WORK, ignore_errors=True)
    data, truth, docs = (os.path.join(WORK, d) for d in ("data", "truth", "docs"))
    for d in (data, truth, docs):
        os.makedirs(d)
    subprocess.run([sys.executable, "-I", os.path.join(HERE, "walk-tables.py"), data, truth, "windows"], check=True)
    for ko, ascii_name in ((v, k) for k, v in KO_RENAMES.items()):  # the zip carries ASCII names; the machine renames them
        os.replace(os.path.join(data, ko), os.path.join(data, ascii_name))
    tag = "v" + args.published
    docs_names = ["README.md"] + [os.path.basename(f) for f in git(["ls-tree", "--name-only", tag, "docs/"]).split() if f.endswith(".md")]
    for name in docs_names:
        path = "README.md" if name == "README.md" else "docs/" + name
        with open(os.path.join(docs, name), "wb") as f:
            f.write(git(["show", tag + ":" + path], binary=True))
    with open(os.path.join(WORK, "VERSION"), "w") as f:
        f.write(args.published + "\n")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for folder, prefix in ((data, "data/"), (docs, "docs/")):
            for name in sorted(os.listdir(folder)):
                z.write(os.path.join(folder, name), prefix + name)
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    m = args.machine
    # A machine that already has the tool for its own use keeps it: `down` removes a global tool only when the round put it there.
    had_tool = execute(m, "dotnet tool list --global | findstr /b /i mloop", 60).get("exit_code") == 0
    must(m, "rmdir /s /q \"%s\" 2>nul & mkdir \"%s\\work\" & mkdir \"%s\\with space\"" % (ROOT, ROOT, ROOT), "making the work folder")
    must(m, "echo %s>\"%s\\global-tool-before.txt\"" % ("yes" if had_tool else "no", ROOT), "noting the tool already on the machine")
    chunk = 6000  # a cmd.exe command line holds 8191 characters
    for i in range(0, len(b64), chunk):
        must(m, "echo %s>>\"%s\\in.b64\"" % (b64[i:i + chunk], ROOT), "sending a part of the tables")
    must(m, "certutil -f -decode \"%s\\in.b64\" \"%s\\in.zip\" >nul" % (ROOT, ROOT), "decoding the tables")
    # tar ships with Windows; Expand-Archive of Windows PowerShell 5.1 fails on some current builds ("Cannot find an overload for ExtractToDirectory")
    must(m, "tar -xf \"%s\\in.zip\" -C \"%s\" & del \"%s\\in.zip\" \"%s\\in.b64\"" % (ROOT, ROOT, ROOT, ROOT), "unpacking the tables")
    extract = "$r = Join-Path $env:USERPROFILE 'mloop-walk'"
    if args.ko_folder:
        moves = "; ".join("Move-Item (Join-Path $r 'data\\%s') (Join-Path $r '%s\\%s')" % (src, KO_FOLDER, dst) for src, dst in KO_RENAMES.items())
        extract += "; New-Item -ItemType Directory (Join-Path $r '%s') | Out-Null; %s" % (KO_FOLDER, moves)
    must(m, powershell(extract), "placing the tables")
    listing = must(m, "dir /s /b \"%s\\data\" \"%s\\docs\" | find /c \"\\\"" % (ROOT, ROOT), "counting files")
    print("ready: %s holds data (%d tables) and docs of %s; no mloop is installed" % (ROOT, len(os.listdir(data)), args.published))
    print("files on the machine:", (listing.get("stdout") or "").strip())


def git(args, binary=False):
    out = subprocess.run(["git", "-C", REPO] + args, check=True, capture_output=True).stdout
    return out if binary else out.decode("utf-8")


def down(args):
    m = args.machine
    # Only the processes started from the round's folder are stopped, and the global tool is removed only if the round installed it.
    before = execute(m, "type \"%s\\global-tool-before.txt\"" % ROOT, 60)
    keep_tool = "yes" in (before.get("stdout") or "")
    stop = ("powershell -NoProfile -Command \"Get-Process | Where-Object { $_.Path -like '*\\mloop-walk\\*' } | Stop-Process -Force\"")
    uninstall = "echo kept: the machine had the tool before the round" if keep_tool else "dotnet tool uninstall --global mloop 2>&1"
    show(execute(m, "%s & %s & cd /d %%USERPROFILE%% & rmdir /s /q mloop-walk & echo removed" % (stop, uninstall), 120))
    shutil.rmtree(WORK, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("up")
    u.add_argument("--published", required=True)
    u.add_argument("--machine", required=True)
    u.add_argument("--ko-folder", action="store_true", help="also put the CP949 spreadsheet tables in a folder with a Korean name")
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
    else:
        words = args.command[1:] if args.command[:1] == ["--"] else args.command
        text = " ".join(words)
        sys.exit(show(execute(args.machine, powershell(text) if args.ps else text, args.timeout)))


if __name__ == "__main__":
    main()
