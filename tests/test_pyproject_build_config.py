from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def test_scikit_build_uses_supported_cmake_version_key():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    scikit_build = pyproject["tool"]["scikit-build"]

    assert scikit_build["cmake"]["version"] == ">=3.16"
    assert "minimum-version" not in scikit_build.get("cmake", {})
