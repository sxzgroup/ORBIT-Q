#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

# Define the root path and subdirectories
ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "prompts" / "framework_prompt_template.md"
OUTPUT_DIR = ROOT / "prompts" / "frameworks"

# Supported framework prompts. Add entries here only after the verifier and
# environment policy for that framework are implemented.
FRAMEWORKS = {
    "tensorcircuit": "TensorCircuit-NG or tensorcircuit-nightly",
    "pennylane": "PennyLane",
    "mindquantum": "MindQuantum",
    "torchquantum": "TorchQuantum",
}


def generate_prompt(framework: str, output: Path | None = None) -> Path:
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Template file not found at: {TEMPLATE_PATH}")

    template_content = TEMPLATE_PATH.read_text()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clean_key = framework.lower().strip()
    if clean_key not in FRAMEWORKS:
        supported = ", ".join(sorted(FRAMEWORKS))
        raise ValueError(f"Unsupported framework '{framework}'. Supported: {supported}")

    display_name = FRAMEWORKS[clean_key]
    output_content = template_content.format(framework=display_name)

    output_file = output or (OUTPUT_DIR / f"{clean_key}.md")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(output_content)
    return output_file


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one framework-specific prompt file from template."
    )
    parser.add_argument(
        "--framework",
        nargs="+",
        default="tensorcircuit",
        choices=sorted(FRAMEWORKS),
        help="Framework prompt(s) to generate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path. Defaults to prompts/frameworks/<framework>.md.",
    )
    args = parser.parse_args()

    try:
        if args.output and len(args.framework) != 1:
            raise ValueError(
                "--output can only be used with a single --framework value"
            )
        for framework in args.framework:
            output_file = generate_prompt(framework, args.output)
            display_name = FRAMEWORKS[framework]
            print(
                f"Generated framework prompt: {_display_path(output_file)} "
                f"(Framework: {display_name})"
            )
    except Exception as e:
        print(f"Error generating framework prompts: {e}")
        exit(1)


if __name__ == "__main__":
    main()
