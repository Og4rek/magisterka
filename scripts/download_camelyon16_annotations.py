"""Download only the small CAMELYON16 XML annotations from the public AWS bucket.

The script deliberately does not download whole-slide images or multi-gigabyte
mask TIFFs.  XML polygons are sufficient to rasterize masks for existing PCam
patches when combined with the official metadata coordinates.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path


BUCKET_ENDPOINT = "https://camelyon-dataset.s3.us-west-2.amazonaws.com"
PREFIX = "CAMELYON16/annotations/"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/camelyon16/annotations"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Download files that already exist instead of retaining them.",
    )
    return parser.parse_args()


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def list_annotation_keys() -> list[tuple[str, int, str]]:
    continuation: str | None = None
    objects: list[tuple[str, int, str]] = []
    while True:
        query = {"list-type": "2", "prefix": PREFIX}
        if continuation:
            query["continuation-token"] = continuation
        url = f"{BUCKET_ENDPOINT}/?{urllib.parse.urlencode(query)}"
        with urllib.request.urlopen(url, timeout=60) as response:
            root = ET.fromstring(response.read())
        for content in (item for item in root.iter() if _tag(item) == "Contents"):
            values = {_tag(child): child.text or "" for child in content}
            key = values.get("Key", "")
            if key.lower().endswith(".xml"):
                objects.append(
                    (
                        key,
                        int(values.get("Size", "0")),
                        values.get("ETag", "").strip('"'),
                    )
                )
        truncated = next(
            (item.text for item in root.iter() if _tag(item) == "IsTruncated"),
            "false",
        )
        if truncated.lower() != "true":
            break
        continuation = next(
            (
                item.text
                for item in root.iter()
                if _tag(item) == "NextContinuationToken"
            ),
            None,
        )
        if not continuation:
            raise RuntimeError("S3 listing is truncated without a continuation token.")
    if not objects:
        raise RuntimeError(f"No XML files found under s3://camelyon-dataset/{PREFIX}")
    return objects


def _download(
    item: tuple[str, int, str],
    output: Path,
    overwrite: bool,
) -> dict[str, object]:
    key, expected_size, etag = item
    destination = output / Path(key).name
    if destination.is_file() and destination.stat().st_size == expected_size and not overwrite:
        return {
            "key": key,
            "file": destination.name,
            "size": expected_size,
            "etag": etag,
            "status": "retained",
        }
    url = f"{BUCKET_ENDPOINT}/{urllib.parse.quote(key, safe='/')}"
    temporary = destination.with_suffix(".xml.part")
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    if len(payload) != expected_size:
        raise IOError(
            f"Size mismatch for {key}: expected {expected_size}, received {len(payload)}"
        )
    ET.fromstring(payload)  # Reject HTML errors or malformed downloads.
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return {
        "key": key,
        "file": destination.name,
        "size": expected_size,
        "etag": etag,
        "md5": hashlib.md5(payload).hexdigest(),  # noqa: S324 - integrity only
        "status": "downloaded",
    }


def main() -> None:
    args = parse_arguments()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive.")
    args.output.mkdir(parents=True, exist_ok=True)
    objects = list_annotation_keys()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = list(
            executor.map(
                lambda item: _download(item, args.output, args.overwrite),
                objects,
            )
        )
    manifest = {
        "source": f"s3://camelyon-dataset/{PREFIX}",
        "downloaded_at": datetime.now(UTC).isoformat(),
        "file_count": len(records),
        "total_bytes": sum(int(record["size"]) for record in records),
        "files": records,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    downloaded = sum(record["status"] == "downloaded" for record in records)
    print(
        f"annotations={len(records)} downloaded={downloaded} "
        f"retained={len(records) - downloaded} output={args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
