#!/usr/bin/env python3
"""Run an unchanged canonical evaluator, with paired MPS fixtures where needed."""
import argparse
import hashlib
import importlib.util
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["fixture", "evaluate"])
    parser.add_argument("task", type=int)
    args = parser.parse_args()
    n = args.task
    random.seed(20260905 + n)
    np.random.seed(20260905 + n)
    path = Path(f"/evaluator/evaluate_{n}.py")
    spec = importlib.util.spec_from_file_location(f"evaluate_{n}", path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    config = dict(evaluator.DEFAULT_CONFIG)

    if args.phase == "fixture":
        import quimb as qu
        qu.seed_rand(20260905 + n)
        if n == 1:
            value = evaluator.dmrg_initial_state(config)
        elif n == 12:
            value = evaluator.dmrg_target_state(config)
        else:
            raise ValueError("Only tasks 01 and 12 have evaluator-prepared MPS inputs")
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        Path("/output/fixture.pkl").write_bytes(payload)
        state = value[0]
        manifest = {
            "task": n, "seed": 20260905 + n, "config": config,
            "fixture_sha256": hashlib.sha256(payload).hexdigest(),
            "tensor_sha256": [hashlib.sha256(np.asarray(t.data).tobytes()).hexdigest()
                              for t in state.tensors],
            "preparation": "Unmodified canonical evaluator function; shared by both implementations; outside artifact timer",
        }
        Path("/output/fixture.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"phase": "fixture", "task": n, "sha256": manifest["fixture_sha256"]}), flush=True)
        return

    if n in [1, 12]:
        # Each process unpickles its own identical input, so mutations cannot leak.
        value = pickle.loads(Path("/fixture/fixture.pkl").read_bytes())
        if n == 1:
            evaluator.dmrg_initial_state = lambda cfg: value
        else:
            evaluator.dmrg_target_state = lambda cfg: value
    sys.path.insert(0, "/candidate")
    # Use the official CLI entry point, including task-specific input preparation.
    official_evaluate = evaluator.evaluate

    def record_config_and_evaluate(name, effective_config):
        encoded = json.dumps(effective_config, sort_keys=True,
                             default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item())
        Path("/output/effective-config.json").write_text(encoded + "\n")
        return official_evaluate(name, effective_config)

    evaluator.evaluate = record_config_and_evaluate
    sys.argv = [str(path), "--solution", f"solution_{n}"]
    evaluator.main()


if __name__ == "__main__":
    main()
