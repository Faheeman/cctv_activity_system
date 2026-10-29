"""Test suite package with 9p readdir cache reconciliation for unittest discover."""

import importlib.abc
import importlib.util
import os
import sys

_ORIG_LISTDIR = os.listdir
_KNOWN_EXTRA_FILES = {
    "unit": [
        "test_config.py",
        "test_video_source.py",
        "test_pose_estimator.py",
        "test_renderer.py",
        "test_person_tracker.py",
        "test_activity_classifier.py",
    ],
    "integration": [
        "test_video_ingestion.py",
        "test_pose_pipeline.py",
        "test_activity_evaluation.py",
        "test_end_to_end.py",
        "test_api.py",
    ],
    "fixtures": [
        "video_generator.py",
        "activity_sequences.py",
    ],
}


def _patched_listdir(path="."):
    entries = list(_ORIG_LISTDIR(path))
    try:
        abs_p = os.path.abspath(os.fspath(path))
        base_name = os.path.basename(abs_p)
        if base_name in _KNOWN_EXTRA_FILES:
            for candidate in _KNOWN_EXTRA_FILES[base_name]:
                if candidate not in entries and os.path.isfile(os.path.join(abs_p, candidate)):
                    entries.append(candidate)
    except Exception:
        pass
    return entries


os.listdir = _patched_listdir


class _DirectStatPathFinder(importlib.abc.MetaPathFinder):
    """Fallback MetaPathFinder using direct os.path.isfile(stat) when 9p caches os.listdir."""

    def find_spec(self, fullname, path=None, target=None):
        mod_name = fullname.split(".")[-1]
        search_paths = path if path is not None else sys.path
        for base in search_paths:
            base_dir = os.path.abspath(base if base else os.getcwd())
            if not base_dir.startswith("/working_dir"):
                continue
            pkg_init = os.path.join(base_dir, mod_name, "__init__.py")
            if os.path.isfile(pkg_init):
                return importlib.util.spec_from_file_location(
                    fullname,
                    pkg_init,
                    submodule_search_locations=[os.path.join(base_dir, mod_name)],
                )
            py_file = os.path.join(base_dir, f"{mod_name}.py")
            if os.path.isfile(py_file):
                return importlib.util.spec_from_file_location(fullname, py_file)
        return None


if not any(f.__class__.__name__ == "_DirectStatPathFinder" for f in sys.meta_path):
    sys.meta_path.append(_DirectStatPathFinder())
