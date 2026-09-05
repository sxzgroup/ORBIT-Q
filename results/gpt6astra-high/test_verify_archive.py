"""Regression tests for offline verification; all mutations use disposable copies."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import verify_archive as verifier


class ArchiveTests(unittest.TestCase):
    @contextmanager
    def copy(self):
        with tempfile.TemporaryDirectory(prefix="astra-archive-test-") as directory:
            root = Path(directory) / "archive"
            shutil.copytree(verifier.ROOT, root, ignore=shutil.ignore_patterns("__pycache__"))
            yield root

    def reseal(self, root):
        """Bypass checksums in fixtures to exercise semantic checks as well."""
        path = root / "archive-manifest.json"
        manifest = json.loads(path.read_text())
        manifest["sha256"] = verifier.inventory(root)
        path.write_text(json.dumps(manifest))

    def edit_json(self, root, relative, change):
        path = root / relative
        data = json.loads(path.read_text())
        change(data)
        path.write_text(json.dumps(data))
        self.reseal(root)

    def test_relocated_stdlib_only_read_only(self):
        with self.copy() as root:
            before = verifier.inventory(root)
            env = dict(os.environ, PATH="", PYTHONDONTWRITEBYTECODE="1")
            result = subprocess.run([sys.executable, "-B", "-S", str(root / "verify_archive.py")],
                                    cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"tasks": 12', result.stdout)
            self.assertEqual(before, verifier.inventory(root))

    def test_missing_task(self):
        with self.copy() as root:
            (root / "challenge-12").rename(root / "omitted-task")
            self.reseal(root)
            with self.assertRaisesRegex(verifier.VerificationError, "twelve challenge"):
                verifier.verify(root)

    def test_unrecorded_candidate_change(self):
        with self.copy() as root:
            path = root / "challenge-01/solution_1.py"
            path.write_text(path.read_text() + "\n# changed\n")
            with self.assertRaisesRegex(verifier.VerificationError, "checksum mismatch"):
                verifier.verify(root)

    def test_candidate_change_with_resealed_manifest(self):
        with self.copy() as root:
            path = root / "challenge-01/solution_1.py"
            path.write_text(path.read_text() + "\n# changed\n")
            self.reseal(root)
            with self.assertRaisesRegex(verifier.VerificationError, "candidate hash"):
                verifier.verify(root)

    def test_reward_mismatch(self):
        with self.copy() as root:
            self.edit_json(root, "challenge-01/reward.json", lambda d: d.update(reward=0))
            with self.assertRaisesRegex(verifier.VerificationError, "verifier reward"):
                verifier.verify(root)

    def test_wrong_solver_effort(self):
        with self.copy() as root:
            for name in ["trial-result.json", "trial-result.raw.txt"]:
                path = root / "challenge-01" / name
                path.write_text(path.read_text().replace('"reasoning_effort": "high"',
                                                        '"reasoning_effort": "low"'))
            self.reseal(root)
            with self.assertRaisesRegex(verifier.VerificationError, "solver model/effort"):
                verifier.verify(root)

    def test_token_sum_mismatch(self):
        with self.copy() as root:
            self.edit_json(root, "comparison.json", lambda d: d["models"]["astra_high"].update(total_tokens=1))
            with self.assertRaisesRegex(verifier.VerificationError, "total total_tokens"):
                verifier.verify(root)

    def test_container_image_mismatch(self):
        with self.copy() as root:
            path = "container-evidence/challenge-01__t5v3pbq__env-main-1.json"
            self.edit_json(root, path, lambda d: d.update(image="wrong-image"))
            with self.assertRaisesRegex(verifier.VerificationError, "actual container resources"):
                verifier.verify(root)

    def test_historical_comparison_mismatch(self):
        with self.copy() as root:
            self.edit_json(root, "public-model-comparison.json", lambda d: d["runs"][0].update(total_cost_usd=1))
            with self.assertRaisesRegex(verifier.VerificationError, "Pinned PR #28"):
                verifier.verify(root)

    def test_missing_figure(self):
        with self.copy() as root:
            (root / "figs/astra-high-agent-resource-use.svg").rename(root / "omitted-figure.svg")
            self.reseal(root)
            with self.assertRaisesRegex(verifier.VerificationError, "exactly two figures"):
                verifier.verify(root)

    def test_corrupt_png(self):
        with self.copy() as root:
            path = root / "figs/astra-high-agent-resource-use.png"
            contents = bytearray(path.read_bytes())
            contents[29] ^= 1  # IHDR CRC, not image dimensions.
            path.write_bytes(contents)
            self.reseal(root)
            with self.assertRaisesRegex(verifier.VerificationError, "PNG CRC"):
                verifier.verify(root)

    def test_reject_nonfinite_json(self):
        with self.assertRaises(verifier.VerificationError):
            verifier.parse_json('{"cost": NaN}')

    def test_runtime_estimate_not_measurement(self):
        with self.copy() as root:
            self.edit_json(root, "runtime/astra_expert_adjustment.json",
                           lambda d: d["summary"].update(measured_pair_count=12))
            with self.assertRaisesRegex(verifier.VerificationError, "measured_pair_count"):
                verifier.verify(root)

    def test_wrong_runtime_reference(self):
        with self.copy() as root:
            self.edit_json(root, "runtime/astra_expert_adjustment.json",
                           lambda d: d["task08"].update(expert_reference_sec=25.05))
            with self.assertRaisesRegex(verifier.VerificationError, "Approved Task 08 reference"):
                verifier.verify(root)

    def test_stale_runtime_aggregate(self):
        with self.copy() as root:
            self.edit_json(root, "runtime/astra_expert_adjustment.json",
                           lambda d: d["summary"].update(geometric_mean_ratio=0.564975))
            with self.assertRaisesRegex(verifier.VerificationError, "geometric_mean_ratio"):
                verifier.verify(root)

    def test_reject_duplicate_json_keys(self):
        with self.assertRaisesRegex(verifier.VerificationError, "Duplicate JSON key"):
            verifier.parse_json('{"reward": 0, "reward": 1}')


if __name__ == "__main__":
    unittest.main()
