"""Map PCam metadata coordinates to CAMELYON16 XML tumor annotations."""

from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True, slots=True)
class PCamMetadataRecord:
    sample_id: int
    coord_y: float
    coord_x: float
    tumor_patch: bool
    center_tumor_patch: bool
    wsi: str


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


class PCamMetadata:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        records: list[PCamMetadataRecord] = []
        with self.path.open(encoding="utf-8-sig", newline="") as handle:
            for fallback_id, row in enumerate(csv.DictReader(handle)):
                sample_text = row.get("") or row.get("sample_id") or str(fallback_id)
                records.append(
                    PCamMetadataRecord(
                        sample_id=int(sample_text),
                        coord_y=float(row["coord_y"]),
                        coord_x=float(row["coord_x"]),
                        tumor_patch=_as_bool(row["tumor_patch"]),
                        center_tumor_patch=_as_bool(row["center_tumor_patch"]),
                        wsi=row["wsi"],
                    )
                )
        self.records = records
        self._by_id = {record.sample_id: record for record in records}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, sample_id: int) -> PCamMetadataRecord:
        return self._by_id[int(sample_id)]


@dataclass(frozen=True, slots=True)
class AnnotationPolygons:
    positive: tuple[tuple[tuple[float, float], ...], ...]
    negative: tuple[tuple[tuple[float, float], ...], ...]


def _local_tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


@lru_cache(maxsize=256)
def parse_camelyon_xml(path_text: str) -> AnnotationPolygons:
    root = ET.parse(path_text).getroot()
    positive: list[tuple[tuple[float, float], ...]] = []
    negative: list[tuple[tuple[float, float], ...]] = []
    for annotation in root.iter():
        if _local_tag(annotation) != "Annotation":
            continue
        coordinates = [
            (float(point.attrib["X"]), float(point.attrib["Y"]))
            for point in annotation.iter()
            if _local_tag(point) == "Coordinate"
            and "X" in point.attrib
            and "Y" in point.attrib
        ]
        if len(coordinates) < 3:
            continue
        group = annotation.attrib.get("PartOfGroup", "").strip().lower()
        is_negative = group in {"_2", "negative", "exclusion", "exclude"}
        (negative if is_negative else positive).append(tuple(coordinates))
    return AnnotationPolygons(tuple(positive), tuple(negative))


def annotation_stem(wsi: str) -> str:
    normalized = wsi.strip().lower()
    normalized = re.sub(r"^camelyon16_(train|training)_", "", normalized)
    normalized = re.sub(r"^camelyon16_", "", normalized)
    return normalized


class CamelyonMaskProvider:
    """Rasterize a 96x96 tumor mask from level-0 annotation polygons."""

    def __init__(
        self,
        annotations_directory: str | Path,
        patch_size: int = 96,
        downsample: float = 4.0,
        coordinate_mode: str = "top_left",
    ) -> None:
        if coordinate_mode not in {"top_left", "center"}:
            raise ValueError("coordinate_mode must be top_left or center.")
        self.annotations_directory = Path(annotations_directory)
        self.patch_size = int(patch_size)
        self.downsample = float(downsample)
        self.coordinate_mode = coordinate_mode
        self._paths = {
            path.stem.lower(): path
            for path in self.annotations_directory.glob("*.xml")
        }

    def annotation_path(self, wsi: str) -> Path | None:
        return self._paths.get(annotation_stem(wsi))

    def _origin(self, record: PCamMetadataRecord) -> tuple[float, float]:
        x, y = record.coord_x, record.coord_y
        if self.coordinate_mode == "center":
            half_extent = self.patch_size * self.downsample / 2
            x -= half_extent
            y -= half_extent
        return x, y

    def mask(self, record: PCamMetadataRecord) -> tuple[np.ndarray, bool]:
        path = self.annotation_path(record.wsi)
        if path is None:
            return np.zeros((self.patch_size, self.patch_size), dtype=np.uint8), False
        polygons = parse_camelyon_xml(str(path.resolve()))
        origin_x, origin_y = self._origin(record)
        canvas = Image.new("L", (self.patch_size, self.patch_size), color=0)
        draw = ImageDraw.Draw(canvas)

        def project(points: tuple[tuple[float, float], ...]):
            return [
                (
                    (x - origin_x) / self.downsample,
                    (y - origin_y) / self.downsample,
                )
                for x, y in points
            ]

        for polygon in polygons.positive:
            draw.polygon(project(polygon), fill=1)
        for polygon in polygons.negative:
            draw.polygon(project(polygon), fill=0)
        return np.asarray(canvas, dtype=np.uint8), True

    @staticmethod
    def center_contains_tumor(mask: np.ndarray) -> bool:
        height, width = mask.shape
        top, bottom = height // 3, 2 * height // 3
        left, right = width // 3, 2 * width // 3
        return bool(mask[top:bottom, left:right].any())
