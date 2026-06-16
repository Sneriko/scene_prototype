"""PEP 660 editable build backend for this project.

The standard setuptools backend is still used for regular wheel/sdist builds.
Editable installs are implemented here so older setuptools installations do not
fall back to the deprecated ``setup.py develop`` path.
"""

from __future__ import annotations

import base64
import csv
import hashlib
from pathlib import Path
import zipfile


_DIST_INFO = "ambulance_case_backend-0.1.0.dist-info"
_WHEEL_NAME = "ambulance_case_backend-0.1.0-py3-none-any.whl"
_PTH_NAME = "ambulance_case_backend_editable.pth"

_METADATA = """Metadata-Version: 2.1
Name: ambulance-case-backend
Version: 0.1.0
Summary: Backend pipeline for ambulance audio transcription, treatment support, and journal drafting.
Requires-Python: >=3.10
Requires-Dist: openai>=1.30.0
Requires-Dist: pydantic>=2.6.0
Requires-Dist: pypdf>=4.2.0
Requires-Dist: python-dotenv>=1.0.1
Provides-Extra: dev
Requires-Dist: pytest>=8.0.0; extra == \"dev\"
Provides-Extra: local-asr
Requires-Dist: transformers>=4.41.0; extra == \"local-asr\"
Requires-Dist: torch>=2.2.0; extra == \"local-asr\"
Requires-Dist: pyannote.audio>=3.1.0; extra == \"local-asr\"
Provides-Extra: edge
Requires-Dist: fastapi>=0.111.0; extra == \"edge\"
Requires-Dist: uvicorn[standard]>=0.29.0; extra == \"edge\"
Requires-Dist: python-multipart>=0.0.9; extra == \"edge\"
"""

_WHEEL = """Wheel-Version: 1.0
Generator: scene-prototype-editable-backend
Root-Is-Purelib: true
Tag: py3-none-any
"""

_ENTRY_POINTS = """[console_scripts]
ambulance-case = ambulance_case_backend.cli:main
"""


def _supported_features():
    return ["build_editable"]


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _setuptools_backend().build_wheel(
        wheel_directory, config_settings=config_settings, metadata_directory=metadata_directory
    )


def build_sdist(sdist_directory, config_settings=None):
    return _setuptools_backend().build_sdist(sdist_directory, config_settings=config_settings)


def get_requires_for_build_wheel(config_settings=None):
    return _setuptools_backend().get_requires_for_build_wheel(config_settings=config_settings)


def get_requires_for_build_sdist(config_settings=None):
    return _setuptools_backend().get_requires_for_build_sdist(config_settings=config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    return _setuptools_backend().prepare_metadata_for_build_wheel(
        metadata_directory, config_settings=config_settings
    )


def get_requires_for_build_editable(config_settings=None):
    return []


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    dist_info = Path(metadata_directory) / _DIST_INFO
    _write_dist_info(dist_info)
    return _DIST_INFO


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    wheel_path = Path(wheel_directory) / _WHEEL_NAME
    source_root = (Path(__file__).resolve().parent / "src").as_posix()

    files = {
        _PTH_NAME: f"{source_root}\n".encode(),
        f"{_DIST_INFO}/METADATA": _METADATA.encode(),
        f"{_DIST_INFO}/WHEEL": _WHEEL.encode(),
        f"{_DIST_INFO}/entry_points.txt": _ENTRY_POINTS.encode(),
    }

    records = []
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for path, content in files.items():
            wheel.writestr(path, content)
            records.append((path, _hash(content), str(len(content))))

        record_path = f"{_DIST_INFO}/RECORD"
        record_content = _record_csv(records + [(record_path, "", "")])
        wheel.writestr(record_path, record_content)

    return _WHEEL_NAME


def _setuptools_backend():
    from setuptools import build_meta

    return build_meta


def _write_dist_info(dist_info):
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(_METADATA)
    (dist_info / "WHEEL").write_text(_WHEEL)
    (dist_info / "entry_points.txt").write_text(_ENTRY_POINTS)


def _hash(content):
    digest = hashlib.sha256(content).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}"


def _record_csv(rows):
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue().encode()
