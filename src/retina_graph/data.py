from __future__ import annotations

from pathlib import Path
from urllib.request import Request

from .network import open_verified_url
from zipfile import ZipFile, BadZipFile
import re
import shutil
import tempfile

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .hrf_data import ensure_hrf_data, iter_hrf_samples, RetinalSample

CHASE_URL = "https://researchinnovation.kingston.ac.uk/files/40659508/CHASEDB1.zip"
CHASE_MIRROR_URL = "https://raw.githubusercontent.com/playyalp1/CHASE_DB1/main/CHASEDB1.zip"
CHASE_SOURCE = "https://researchinnovation.kingston.ac.uk/en/datasets/chasedb1-retinal-vessel-reference-dataset-4/"
CHASE_MIRROR_SOURCE = "https://github.com/playyalp1/CHASE_DB1"


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "retinal-vascular-graph-stability/1.0"})
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with open_verified_url(request, timeout=90) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if not temporary.exists() or temporary.stat().st_size == 0:
            raise RuntimeError("The dataset download returned an empty file.")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with ZipFile(archive) as zf:
        for member in zf.infolist():
            resolved = (target / member.filename).resolve()
            if root != resolved and root not in resolved.parents:
                raise RuntimeError("Unsafe path detected in dataset archive.")
        zf.extractall(target)


def _valid_chase_root(root: Path) -> bool:
    if not root.exists():
        return False
    images = [p for p in root.rglob("*.jpg") if re.fullmatch(r"Image_\d+[LR]", p.stem, flags=re.I)]
    masks = [p for p in root.rglob("*.png") if re.fullmatch(r"Image_\d+[LR]_1stHO", p.stem, flags=re.I)]
    return len(images) == 28 and len(masks) == 28


def ensure_chase_data(project_root: Path | str) -> Path:
    project_root = Path(project_root).resolve()
    root = project_root / "data" / "CHASE_DB1"
    if _valid_chase_root(root):
        return root
    raw = project_root / "data" / "raw"
    archive = raw / "chase_db1.zip"
    if not archive.exists() or archive.stat().st_size == 0:
        errors = []
        sources = ((CHASE_URL, CHASE_SOURCE), (CHASE_MIRROR_URL, CHASE_MIRROR_SOURCE))
        for url, label in sources:
            print(f"Downloading CHASE_DB1 from {label}")
            try:
                _download(url, archive)
                with ZipFile(archive) as zf:
                    bad_member = zf.testzip()
                    if bad_member:
                        raise BadZipFile(f"Corrupt archive member: {bad_member}")
                    names = [item.filename for item in zf.infolist()]
                    image_count = sum(bool(re.search(r"Image_\d+[LR]\.jpg$", name, flags=re.I)) for name in names)
                    mask_count = sum(bool(re.search(r"Image_\d+[LR]_1stHO\.png$", name, flags=re.I)) for name in names)
                    if image_count != 28 or mask_count != 28:
                        raise RuntimeError(
                            f"Archive contents did not match CHASE_DB1: found {image_count} images "
                            f"and {mask_count} first-observer masks; expected 28 of each."
                        )
                break
            except Exception as exc:
                errors.append(f"{label}: {type(exc).__name__}: {exc}")
                archive.unlink(missing_ok=True)
        else:
            details = "\n".join(errors)
            raise RuntimeError(
                "CHASE_DB1 could not be downloaded from the official Kingston source or the "
                "public mirror. This does not affect the HRF dataset, which is already available.\n"
                "Continue with HRF only using `python scripts\\download_data.py --hrf-only`.\n"
                "To add CHASE_DB1 later, download the archive manually from either source listed "
                "in the README, place it at data\\raw\\chase_db1.zip, and rerun this script.\n"
                f"Download errors:\n{details}"
            )

    try:
        with ZipFile(archive) as zf:
            bad_member = zf.testzip()
            if bad_member:
                raise BadZipFile(f"Corrupt archive member: {bad_member}")
    except BadZipFile as exc:
        archive.unlink(missing_ok=True)
        raise RuntimeError("The CHASE_DB1 archive is invalid. Run the downloader again.") from exc
    with tempfile.TemporaryDirectory(prefix="chase_extract_", dir=raw) as temp:
        extracted = Path(temp)
        _safe_extract(archive, extracted)
        if not _valid_chase_root(extracted):
            raise RuntimeError("Expected 28 CHASE_DB1 images and 28 first-observer masks in the Kingston archive.")
        if root.exists():
            shutil.rmtree(root)
        shutil.copytree(extracted, root)
    if not _valid_chase_root(root):
        raise RuntimeError("CHASE_DB1 validation failed after extraction.")
    return root


def _estimate_fov(image: np.ndarray) -> np.ndarray:
    foreground = np.any(image > 8, axis=2)
    labels, count = ndi.label(foreground, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.ones(image.shape[:2], dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = labels == int(np.argmax(sizes))
    return ndi.binary_fill_holes(largest)


def iter_all_samples(hrf_root: Path | str, chase_root: Path | str | None = None, include_chase: bool = True):
    yield from iter_hrf_samples(hrf_root)
    if not include_chase or chase_root is None:
        return
    chase_root = Path(chase_root)
    image_paths = sorted(p for p in chase_root.rglob("*.jpg") if re.fullmatch(r"Image_\d+[LR]", p.stem, flags=re.I))
    masks = {p.stem.lower(): p for p in chase_root.rglob("*.png") if re.fullmatch(r"Image_\d+[LR]_1stHO", p.stem, flags=re.I)}
    if len(image_paths) != 28:
        raise RuntimeError(f"Expected 28 CHASE_DB1 images, found {len(image_paths)}.")
    for image_path in image_paths:
        key = image_path.stem.lower()
        mask_path = masks.get(f"{key}_1stho")
        if mask_path is None:
            raise FileNotFoundError(f"Missing first-observer vessel annotation for {image_path.stem}.")
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        vessel = np.asarray(Image.open(mask_path).convert("L")) >= 128
        fov = _estimate_fov(image)
        if image.shape[:2] != vessel.shape:
            raise ValueError(f"CHASE_DB1 image and mask dimensions differ for {image_path.stem}.")
        yield RetinalSample("CHASE_DB1", image_path.stem, image, vessel, fov, "not_recorded")
