"""Model-free tests for the candidate atomic result publisher.

Synthetic fixtures only: no model, tokenizer, provider, numpy or torch import,
no real diagnostic rows, no fitting. Each test uses a private temporary
directory. The two-publisher race test spawns two bounded, owned subprocesses
that are always joined (and killed on timeout) before the test returns.
"""
import errno, hashlib, json, os, subprocess, sys, tempfile, time, unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import atomic_result_writer as writer
import diagnostic_scoring as scoring

SAMPLE = {"schema": "prechoice_diagnostic_scoring_result.v1", "correct": 1, "total": 2}
CHILD = (
    "import json, os, sys, time\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "import atomic_result_writer as a\n"
    "dest, sentinel, payload, ready, release = sys.argv[2:7]\n"
    "target = os.path.abspath(dest)\n"
    "real_link = os.link\n"
    "def barrier(src, dst, *args, **kwargs):\n"
    "    if os.path.abspath(os.fspath(dst)) == target:\n"
    "        marker = os.open(ready, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)\n"
    "        os.close(marker)\n"
    "        deadline = time.monotonic() + 20\n"
    "        while not os.path.exists(release):\n"
    "            if time.monotonic() > deadline:\n"
    "                sys.exit(7)\n"
    "            time.sleep(0.001)\n"
    "    return real_link(src, dst, *args, **kwargs)\n"
    "os.link = barrier\n"
    "probe_src = ready + '.probe.src'\n"
    "probe_dst = ready + '.probe.dst'\n"
    "with open(probe_src, 'wb') as handle:\n"
    "    handle.write(b'probe')\n"
    "os.link(probe_src, probe_dst)\n"
    "if os.path.exists(ready):\n"
    "    sys.exit(8)\n"
    "os.unlink(probe_src)\n"
    "os.unlink(probe_dst)\n"
    "deadline = time.monotonic() + 20\n"
    "while not os.path.exists(sentinel):\n"
    "    if time.monotonic() > deadline:\n"
    "        sys.exit(9)\n"
    "    time.sleep(0.001)\n"
    "code = 0\n"
    "try:\n"
    "    a.write_result(json.loads(payload), dest)\n"
    "except ValueError as exc:\n"
    "    print('LOSE', exc)\n"
    "    code = 3\n"
    "else:\n"
    "    print('WIN')\n"
    "finally:\n"
    "    try:\n"
    "        os.unlink(ready)\n"
    "    except OSError:\n"
    "        pass\n"
    "sys.exit(code)\n"
)


def leftovers(directory):
    return sorted(p.name for p in Path(directory).iterdir())


class AtomicResultWriterTests(unittest.TestCase):
    def test_exact_canonical_bytes_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "result.json"
            digest = writer.write_result(SAMPLE, destination)
            raw = destination.read_bytes()
            self.assertEqual(raw, scoring.json_bytes(SAMPLE))
            self.assertEqual(raw, writer.json_bytes(SAMPLE))
            self.assertEqual(digest, hashlib.sha256(raw).hexdigest())
            self.assertEqual(json.loads(raw.decode("ascii")), SAMPLE)
            self.assertEqual(leftovers(tmp), ["result.json"])

    def test_existing_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "result.json"
            destination.write_bytes(b"PREEXISTING\n")
            with self.assertRaises(ValueError) as caught:
                writer.write_result(SAMPLE, destination)
            self.assertEqual(str(caught.exception), "RESULT_ALREADY_EXISTS")
            self.assertEqual(destination.read_bytes(), b"PREEXISTING\n")
            self.assertEqual(leftovers(tmp), ["result.json"])

    def test_oversize_payload_rejected_without_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "result.json"
            big = {"blob": "a" * (writer.OUTPUT_BYTES + 1)}
            self.assertGreater(len(writer.json_bytes(big)), writer.OUTPUT_BYTES)
            with self.assertRaises(ValueError) as caught:
                writer.write_result(big, destination)
            self.assertEqual(str(caught.exception), "SCORE_OUTPUT_BOUND")
            self.assertFalse(destination.exists())
            self.assertEqual(leftovers(tmp), [])

    def test_two_concurrent_publishers_exactly_one_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "result.json"
            start = Path(tmp) / "start"
            release = Path(tmp) / "release"
            ready = [Path(tmp) / "ready.0", Path(tmp) / "ready.1"]
            payload = json.dumps(SAMPLE)
            children = []
            try:
                for marker in ready:
                    command = [
                        sys.executable, "-E", "-S", "-B", "-c", CHILD,
                        str(HERE), str(destination), str(start), payload,
                        str(marker), str(release),
                    ]
                    children.append(
                        subprocess.Popen(
                            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                        )
                    )
                start.write_bytes(b"go")
                # Release the barrier only once BOTH publishers have passed the
                # destination-absent precheck and reached the real os.link call.
                deadline = time.monotonic() + 30
                while not all(marker.exists() for marker in ready):
                    if time.monotonic() > deadline:
                        self.fail("publishers did not both reach the destination os.link barrier")
                    time.sleep(0.002)
                self.assertTrue(all(marker.exists() for marker in ready))
                release.write_bytes(b"go")
                results = []
                for child in children:
                    out, err = child.communicate(timeout=60)
                    results.append((child.returncode, out.strip(), err.strip()))
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill()
                    child.wait()
            self.assertEqual(sorted(code for code, _, _ in results), [0, 3], results)
            outcomes = [out for _, out, _ in results]
            self.assertEqual(sum(out == "WIN" for out in outcomes), 1, results)
            self.assertEqual(
                sum(out == "LOSE RESULT_ALREADY_EXISTS" for out in outcomes), 1, results
            )
            for _, _, err in results:
                self.assertNotIn("Traceback", err, results)
            raw = destination.read_bytes()
            self.assertEqual(raw, scoring.json_bytes(SAMPLE))
            self.assertEqual(json.loads(raw.decode("ascii")), SAMPLE)
            self.assertEqual(writer.sha(raw), hashlib.sha256(raw).hexdigest())
            # Children removed their own barrier markers and temp files.
            self.assertFalse(ready[0].exists(), "child 0 left its barrier marker")
            self.assertFalse(ready[1].exists(), "child 1 left its barrier marker")
            self.assertEqual(sorted(p.name for p in Path(tmp).glob("*.tmp")), [])
            release.unlink()
            self.assertEqual(leftovers(tmp), ["result.json", "start"])

    def test_injected_publish_failure_is_cleaned_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "result.json"
            for number, code in ((errno.EPERM, "RESULT_PUBLISH_UNSUPPORTED"), (errno.EIO, "RESULT_PUBLISH_FAILED")):
                with mock.patch.object(writer.os, "link", side_effect=OSError(number, "injected")):
                    with self.assertRaises(ValueError) as caught:
                        writer.write_result(SAMPLE, destination)
                self.assertEqual(str(caught.exception), code)
                self.assertFalse(destination.exists())
                self.assertEqual(leftovers(tmp), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
