"""Generate deterministic QR_A and QR_B PNG files for the measurement setup."""

from __future__ import annotations

import argparse
from pathlib import Path


PAYLOADS = ("QR_A", "QR_B")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the QR_A and QR_B markers used by this project."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/qr"),
        help="Destination directory (default: outputs/qr)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing QR_A.png and QR_B.png files",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        import qrcode
        from qrcode.constants import ERROR_CORRECT_M
    except ImportError as exc:
        raise SystemExit(
            "qrcode is required; run with: uv run --locked --extra qr "
            "python tools/generate_qr.py"
        ) from exc

    destinations = [args.output_dir / f"{payload}.png" for payload in PAYLOADS]
    existing = [path for path in destinations if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise SystemExit(f"refusing to overwrite existing file(s): {names}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for payload, destination in zip(PAYLOADS, destinations, strict=True):
        qr = qrcode.QRCode(
            version=1,
            error_correction=ERROR_CORRECT_M,
            box_size=20,
            border=4,
        )
        qr.add_data(payload)
        qr.make(fit=False)
        image = qr.make_image(fill_color="black", back_color="white")
        image.save(destination)
        print(f"generated {destination} (payload={payload}, active_modules=21, border=4)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
