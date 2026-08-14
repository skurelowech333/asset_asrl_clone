from pathlib import Path


def test_python_app_workflow_bootstraps_native_build_requirements():
    workflow_path = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "python-app.yml"
    )
    workflow = workflow_path.read_text(encoding="utf-8")

    assert "git submodule sync --recursive" in workflow
    assert "git submodule update --init --recursive" in workflow
    assert "source /opt/intel/oneapi/setvars.sh" in workflow
    assert "pip install --no-build-isolation -r requirements.txt" in workflow
