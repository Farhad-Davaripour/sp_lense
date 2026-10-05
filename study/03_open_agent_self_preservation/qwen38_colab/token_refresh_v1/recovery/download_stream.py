"""Stream a connector file reference to C:, verify, and never print its URL."""
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path


def main():
    item = json.load(sys.stdin)
    destination = Path(item["destination"]).resolve()
    allowed = Path("C:/Users/farha/Research3").resolve()
    if not destination.is_relative_to(allowed):
        raise ValueError("Destination must stay in Research3 on C:")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".recovery.partial")
    digest = hashlib.sha256()
    size = 0
    try:
        request = urllib.request.Request(item["url"], headers={"User-Agent": "Mozilla/5.0", "Accept": "application/octet-stream"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                digest.update(block)
                size += len(block)
        if size != item["bytes"] or digest.hexdigest() != item["sha256"]:
            raise ValueError("Downloaded content failed manifest verification")
        os.replace(temporary, destination)
        print(json.dumps({"name": destination.name, "verified": True, "bytes": size, "sha256": digest.hexdigest()}))
    except Exception as exc:
        print(json.dumps({"name": destination.name, "verified": False, "error_type": type(exc).__name__, "http_status": getattr(exc, "code", None)}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
