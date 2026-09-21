"""Run regression tests with a coherent temporary home and no service credentials."""

from pathlib import Path
import os
import site
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
(root / ".local-state").mkdir(exist_ok=True)
user_site = Path(site.getusersitepackages())
pythonpath = [str(root / "src")]
if user_site.exists():
    pythonpath.append(str(user_site))
with tempfile.TemporaryDirectory(prefix="test-home-", dir=root / ".local-state") as home:
    roaming = Path(home) / "AppData" / "Roaming"
    local = Path(home) / "AppData" / "Local"
    roaming.mkdir(parents=True)
    local.mkdir(parents=True)
    names = (
        "PATH",
        "SystemRoot",
        "SystemDrive",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "CUDA_PATH",
        "CUDA_PATH_V13_0",
        "CUDA_PATH_V12_9",
        "PLAYWRIGHT_BROWSERS_PATH",
        "EUHNN_REQUIRE_CUDA",
    )
    env = {key: os.environ[key] for key in names if key in os.environ}
    env.update(
        {
            # Keep source-checkout tests independent of whether pip used the
            # global or user site-packages directory before HOME was isolated.
            "PYTHONPATH": os.pathsep.join(
                filter(None, [*pythonpath, os.environ.get("PYTHONPATH")])
            ),
            "HOME": home,
            "USERPROFILE": home,
            "TEMP": home,
            "TMP": home,
            "APPDATA": str(roaming),
            "LOCALAPPDATA": str(local),
            "XDG_DATA_HOME": str(local),
            "XDG_CACHE_HOME": str(local),
            "OTEL_SDK_DISABLED": "true",
            "CREWAI_TELEMETRY_DISABLED": "true",
            "CREWAI_TRACING_ENABLED": "false",
            "HF_HUB_OFFLINE": "1",
            "HAYSTACK_TELEMETRY_ENABLED": "False",
            "DO_NOT_TRACK": "1",
            "AGNO_TELEMETRY": "false",
            "LANGCHAIN_TRACING_V2": "false",
            "LANGSMITH_TRACING": "false",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "EUHNN_NATIVE_EVIDENCE": str(root / "audit/native-integrations.json"),
        }
    )
    result = subprocess.run([sys.executable, "-m", "pytest", *(sys.argv[1:] or ["tests", "-q"])], cwd=root, env=env)
raise SystemExit(result.returncode)

