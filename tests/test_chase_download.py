from pathlib import Path
from zipfile import ZipFile

import pytest

from retina_graph import data


def _make_chase_archive(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        for index in range(1, 15):
            for eye in ("L", "R"):
                stem = f"Image_{index:02d}{eye}"
                archive.writestr(f"CHASE/{stem}.jpg", b"image-placeholder")
                archive.writestr(f"CHASE/{stem}_1stHO.png", b"mask-placeholder")


def test_chase_download_uses_mirror_after_official_403(tmp_path, monkeypatch):
    calls = []

    def fake_download(url, destination):
        calls.append(url)
        if url == data.CHASE_URL:
            raise PermissionError("HTTP Error 403: Forbidden")
        destination.parent.mkdir(parents=True, exist_ok=True)
        _make_chase_archive(destination)

    monkeypatch.setattr(data, "_download", fake_download)
    root = data.ensure_chase_data(tmp_path)

    assert calls == [data.CHASE_URL, data.CHASE_MIRROR_URL]
    assert data._valid_chase_root(root)


def test_chase_download_reports_both_sources_unavailable(tmp_path, monkeypatch):
    def fail_download(url, destination):
        raise PermissionError("HTTP Error 403: Forbidden")

    monkeypatch.setattr(data, "_download", fail_download)
    with pytest.raises(RuntimeError, match="CHASE_DB1 could not be downloaded"):
        data.ensure_chase_data(tmp_path)
