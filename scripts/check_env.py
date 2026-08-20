from __future__ import annotations

import argparse
import importlib.util
import platform


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-install", action="store_true")
    args = parser.parse_args()

    print(f"python={platform.python_version()}")
    try:
        import torch

        print(f"torch={torch.__version__}")
        print(f"torch_cuda={torch.version.cuda}")
        print(f"cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"gpu={torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"torch_error={exc!r}")

    if not args.pre_install:
        for name in ("transformers", "minicpmo_utils", "soundfile", "scipy"):
            print(f"{name}_installed={importlib.util.find_spec(name) is not None}")


if __name__ == "__main__":
    main()
