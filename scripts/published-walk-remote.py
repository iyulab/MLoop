#!/usr/bin/env python3
"""The Linux install walk on another machine: a native Linux x64 container in the Docker of a Windows machine reached by command
execution (walk_remote.py). `published-walk.sh ... --machine <name>` calls this; it is not meant to be run by hand.

    published-walk-remote.py up    --machine <name> --work <dir> --container <name> --image <image>
    published-walk-remote.py exec  --machine <name> --container <name> [--timeout <seconds>] -- <bash command>
    published-walk-remote.py fetch --machine <name> --container <name> <container path> <local directory>
    published-walk-remote.py down  --machine <name> --container <name>

`up` copies <dir>/feed, <dir>/data and <dir>/docs into a new container as /feed, /data and /docs. Docker Desktop is started when it is
not running, and `down` stops it again only in that case; no container but the walk's own is touched. `exec` runs a command in a login
shell from /root/work and waits for it, however long it takes (up to --timeout): the transport allows 300 s a call, so the command runs
detached in the container and its output is read back when it ends. A server the command leaves in the background keeps running.
"""
import argparse
import base64
import os
import shutil
import sys
import tarfile
import tempfile
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from walk_remote import download, execute, home, must, run_in_background, upload  # noqa: E402


def folder(machine):
    return home(machine) + "\\mloop-walk-linux"


def up(args):
    m, name = args.machine, args.container
    root = folder(m)
    must(m, 'rmdir /s /q "%s" 2>nul & mkdir "%s"' % (root, root), "making the work folder")
    if execute(m, "docker version >nul 2>&1", 60).get("exit_code") != 0:
        must(m, 'echo yes> "%s\\started-docker.txt"' % root, "noting that the walk starts Docker")
        run_in_background(m, "docker desktop start", root + "\\docker-start.log", "starting Docker Desktop", 600)
    if execute(m, "docker image inspect %s >nul 2>&1" % args.image, 60).get("exit_code") != 0:
        run_in_background(m, "docker pull %s" % args.image, root + "\\pull.log", "pulling the SDK image", 1800)
    with tempfile.TemporaryDirectory() as scratch:
        bundle = os.path.join(scratch, "walk.tar.gz")
        with tarfile.open(bundle, "w:gz") as tar:  # tar keeps the empty folders some of the wrong picture sets consist of
            for part in ("feed", "data", "docs"):
                tar.add(os.path.join(args.work, part), arcname=part)
        upload(m, bundle, root + "\\walk.tar.gz")
    must(m, 'tar -xzf "%s\\walk.tar.gz" -C "%s" && del /q "%s\\walk.tar.gz"' % (root, root, root), "unpacking", 300)
    execute(m, "docker rm -f %s >nul 2>&1" % name, 120)
    must(m, "docker run -d --name %s -e DOTNET_CLI_TELEMETRY_OPTOUT=1 -e DOTNET_NOLOGO=1 %s sleep infinity" % (name, args.image),
         "starting the container", 300)
    for part in ("feed", "data", "docs"):
        must(m, 'docker cp "%s\\%s" %s:%s' % (root, part, name, "/" + part), "copying /%s into the container" % part, 300)
    answer = must(m, 'docker exec %s sh -c "mkdir -p /root/work /tmp/walk && uname -m && dotnet --version && ls /feed /data /docs | head -40"' % name,
                  "checking the container", 120)
    print(answer.get("stdout", ""))


def run(args):
    m, name = args.machine, args.container
    words = args.command[1:] if args.command[:1] == ["--"] else args.command
    text = " ".join(words)
    job = "/tmp/walk/" + uuid.uuid4().hex[:10]
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    if len(encoded) > 6000:
        sys.exit("the command is too long for one call: put it in a script in the container first")
    must(m, 'docker exec %s sh -c "echo %s | base64 -d > %s.sh"' % (name, encoded, job), "sending the command")
    must(m, 'docker exec -d %s bash -c "cd /root/work; bash -l %s.sh > %s.log 2>&1; echo $? > %s.exit"' % (name, job, job, job),
         "starting the command")
    deadline, wait, code = time.time() + args.timeout, 1.0, ""
    while time.time() < deadline:
        time.sleep(wait)
        wait = min(wait * 1.5, 15.0)
        code = execute(m, 'docker exec %s sh -c "cat %s.exit 2>/dev/null"' % (name, job), 60, retries=5).get("stdout", "").strip()
        if code:
            break
    output = execute(m, 'docker exec %s sh -c "tail -c 100000 %s.log"' % (name, job), 120, retries=5).get("stdout", "")
    sys.stdout.write(output)
    if not code:
        sys.stderr.write("[still running after %d s; its output so far is above, the rest goes to %s.log in the container]\n"
                         % (args.timeout, job))
        sys.exit(124)
    sys.exit(int(code))


def fetch(args):
    m, name = args.machine, args.container
    root = folder(m) + "\\fetch"
    must(m, 'rmdir /s /q "%s" 2>nul & mkdir "%s\\out"' % (root, root), "making the fetch folder")
    must(m, 'docker cp %s:%s "%s\\out"' % (name, args.source, root), "copying out of the container", 300)
    must(m, 'tar -czf "%s\\out.tar.gz" -C "%s\\out" .' % (root, root), "packing", 300)
    os.makedirs(args.target, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        bundle = os.path.join(scratch, "out.tar.gz")
        download(m, root + "\\out.tar.gz", bundle)
        with tarfile.open(bundle) as tar:
            tar.extractall(args.target, filter="data")
    print("fetched %s into %s" % (args.source, args.target))


def down(args):
    m, name = args.machine, args.container
    root = folder(m)
    started = "yes" in execute(m, 'type "%s\\started-docker.txt" 2>nul' % root, 60).get("stdout", "")
    execute(m, "docker rm -f %s >nul 2>&1" % name, 120)
    if started:
        run_in_background(m, "docker desktop stop", home(m) + "\\mloop-walk-docker-stop.log", "stopping Docker Desktop", 600)
        execute(m, 'del /q "%USERPROFILE%\\mloop-walk-docker-stop.log" "%USERPROFILE%\\mloop-walk-docker-stop.log.exit" 2>nul', 60)
    execute(m, 'rmdir /s /q "%s"' % root, 120)
    print("removed the walk container and its folder%s" % ("; stopped Docker Desktop, which the walk had started" if started else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for verb in ("up", "exec", "fetch", "down"):
        s = sub.add_parser(verb)
        s.add_argument("--machine", required=True)
        s.add_argument("--container", required=True)
        if verb == "up":
            s.add_argument("--work", required=True)
            s.add_argument("--image", required=True)
        if verb == "exec":
            s.add_argument("--timeout", type=int, default=3600)
            s.add_argument("command", nargs=argparse.REMAINDER)
        if verb == "fetch":
            s.add_argument("source")
            s.add_argument("target")
    args = p.parse_args()
    {"up": up, "exec": run, "fetch": fetch, "down": down}[args.cmd](args)


if __name__ == "__main__":
    main()
