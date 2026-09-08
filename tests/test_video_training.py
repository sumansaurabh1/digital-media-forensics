from __future__ import annotations

import zipfile

from backend.detectors.video_detector.training import discover_videos


def test_zip_discovery_labels_without_extraction(tmp_path) -> None:
    archive_path = tmp_path / "videos.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("real_256/one.mp4", b"real")
        archive.writestr("fake_256/two.mp4", b"fake")

    assert list(discover_videos(archive_path)) == [("fake_256/two.mp4", 1), ("real_256/one.mp4", 0)]
    assert list(tmp_path.iterdir()) == [archive_path]
