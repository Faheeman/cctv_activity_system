"""Real-time human pose estimation, multi-person tracking, and activity recognition."""

import importlib.abc
import importlib.util
import os
import sys

__version__ = "0.1.0"


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
