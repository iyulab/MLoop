"""A Windows machine reached by command execution only - the transport the install walks share.

The machine takes `POST <url>/api/v1/execute` with a JSON body `{"command", "timeout_secs"}` and a bearer key, running the command in
cmd.exe (at most 300 s a call), takes files by chunked upload (`/api/v1/fs/uploads`) and gives them back (`/api/v1/fs/file?path=`).
Its address and key are read from the file named by MLOOP_WALK_WINDOWS_ENV (lines `<name>_url=...` and `<name>_key=...`); neither is
printed or stored anywhere else.
"""
import base64
import hashlib
import json
import os
import re
import ssl
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request


def endpoint(machine):
    path = os.environ.get("MLOOP_WALK_WINDOWS_ENV")
    if not path:
        sys.exit("MLOOP_WALK_WINDOWS_ENV names the file with <name>_url and <name>_key lines; it is not set")
    values = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip().strip('"').strip("'")
    try:
        return values[machine + "_url"].rstrip("/"), values[machine + "_key"]
    except KeyError:
        sys.exit("no %s_url / %s_key in the endpoint file" % (machine, machine))


class TransportError(Exception):
    pass


def _open(machine, method, path, data=None, headers=None, timeout=180):
    url, key = endpoint(machine)
    context = ssl.create_default_context()
    context.check_hostname = False  # the endpoint serves a self-signed certificate
    context.verify_mode = ssl.CERT_NONE
    all_headers = {"Authorization": "Bearer " + key}
    all_headers.update(headers or {})
    request = urllib.request.Request(url + path, data=data, headers=all_headers, method=method)
    try:
        with urllib.request.urlopen(request, context=context, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as e:
        raise TransportError("%s on the machine failed: HTTP %s %s" % (method, e.code, e.read().decode("utf-8", "replace")[:300]))
    except OSError as e:  # the address stays out of the message
        raise TransportError("%s on the machine failed: %s" % (method, type(e).__name__))


def execute(machine, command, timeout=120, retries=0):
    """One command line in cmd.exe; the answer's `exit_code`, `stdout` and `stderr`. `retries` is for commands that only read:
    a call lost on the way is sent again (a command that changes something is never sent twice)."""
    body = json.dumps({"command": command, "timeout_secs": min(timeout, 300)}).encode("utf-8")
    attempt = 0
    while True:
        try:
            answer = _open(machine, "POST", "/api/v1/execute", body, {"Content-Type": "application/json"}, min(timeout, 300) + 30)
            break
        except TransportError as e:
            if attempt >= retries:
                sys.exit(str(e))
            attempt += 1
            time.sleep(5 * attempt)
    return json.loads(answer.decode("utf-8", "replace"))


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
        sys.exit("%s failed (exit %s): %s" % (what, result.get("exit_code"), (result.get("output") or result.get("stdout") or "")[:600]))
    return result


def home(machine):
    """The user profile folder of the account the commands run as."""
    return must(machine, "echo %USERPROFILE%", "reading the profile folder").get("stdout", "").strip()


def upload(machine, local, remote):
    with open(local, "rb") as f:
        blob = f.read()
    start = json.dumps({"path": remote, "size": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}).encode("utf-8")
    try:
        session = json.loads(_open(machine, "POST", "/api/v1/fs/uploads", start, {"Content-Type": "application/json"}))
    except TransportError as e:
        # An upload cut off earlier still holds the path. The paths used here are the walk's own, so its session is the walk's too.
        held = re.search(r'"upload_id":"([^"]+)"', str(e))
        if "destination-busy" not in str(e) or not held:
            raise
        _open(machine, "DELETE", "/api/v1/fs/uploads/%s" % held.group(1))
        session = json.loads(_open(machine, "POST", "/api/v1/fs/uploads", start, {"Content-Type": "application/json"}))
    offset, chunk, upload_id = session.get("offset", 0), session["chunk_size"], session["upload_id"]
    failures = 0
    while offset < len(blob):
        part = blob[offset:offset + chunk]
        try:
            answer = _open(machine, "PATCH", "/api/v1/fs/uploads/%s" % upload_id, part,
                           {"Content-Range": "bytes %d-%d/%d" % (offset, offset + len(part) - 1, len(blob)),
                            "Content-Type": "application/octet-stream"})
            offset, failures = json.loads(answer)["offset"], 0
        except TransportError:
            # A long upload can lose one chunk on the way; the session knows how far it got.
            failures += 1
            if failures > 5:
                raise
            time.sleep(5 * failures)
            offset = json.loads(_open(machine, "GET", "/api/v1/fs/uploads/%s" % upload_id))["offset"]
    _open(machine, "POST", "/api/v1/fs/uploads/%s/complete" % upload_id, b"")


def unpack(archive, target):
    """`archive` into `target`, refusing members that would land outside it where this Python can (3.12, and late 3.9-3.11)."""
    if hasattr(tarfile, "data_filter"):
        archive.extractall(target, filter="data")
        return
    for member in archive.getmembers():
        if member.islnk() or member.issym() or os.path.isabs(member.name) or ".." in member.name.replace("\\", "/").split("/"):
            sys.exit("the archive holds a member that would land outside the target folder: " + member.name)
    archive.extractall(target)


def download(machine, remote, local):
    data = _open(machine, "GET", "/api/v1/fs/file?path=" + urllib.parse.quote(remote), timeout=600)
    with open(local, "wb") as f:
        f.write(data)


def run_in_background(machine, command, log, what, timeout):
    """A cmd.exe command that may outlast one call: started detached, its output and exit code read back from `log`."""
    must(machine, 'del /q "%s" "%s.exit" 2>nul & echo started' % (log, log), "clearing the log of " + what)
    script = "Start-Process cmd -WindowStyle Hidden -ArgumentList '/v:on','/c','%s > \"%s\" 2>&1 & echo !errorlevel! > \"%s.exit\"'" % (
        command.replace("'", "''"), log, log)
    must(machine, powershell(script), "starting " + what)
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(10)
        code = execute(machine, 'type "%s.exit" 2>nul' % log, 60).get("stdout", "").strip()
        if code:
            if code != "0":
                tail = execute(machine, powershell("Get-Content -Tail 30 '%s'" % log), 60).get("stdout", "")
                sys.exit("%s failed (exit %s):\n%s" % (what, code, tail))
            return
    sys.exit("%s did not finish within %d s" % (what, timeout))
