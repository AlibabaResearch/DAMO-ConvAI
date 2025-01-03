import platform
from typer.testing import CliRunner

from sotopia.cli import app
import subprocess
import time

runner = CliRunner()


def test_install() -> None:
    if platform.system() == "Darwin":
        result = runner.invoke(
            app,
            [
                "install",
                "--no-use-docker",
                "--no-load-database",
                "--overwrite-existing-data",
            ],
        )
        assert result.exit_code == 0
        time.sleep(1)
        subprocess.run("redis-cli shutdown", shell=True, check=True)
    elif platform.system() == "Linux":
        result = runner.invoke(
            app,
            [
                "install",
                "--no-use-docker",
                "--load-database",
                "--overwrite-existing-data",
            ],
        )
        assert result.exit_code == 0
        time.sleep(1)
        subprocess.run(
            "./redis-stack-server-7.2.0-v10/bin/redis-cli shutdown",
            shell=True,
            check=True,
        )

    if platform.system() == "Darwin":
        result = runner.invoke(app, ["install"], input="Homebrew\nNo\n")
        assert result.exit_code == 0
        time.sleep(1)
        subprocess.run("redis-cli shutdown", shell=True, check=True)
    elif platform.system() == "Linux":
        result = runner.invoke(app, ["install"], input="Docker\nNo\n\n")
        assert result.exit_code == 0
        time.sleep(1)
        subprocess.run(
            "docker stop redis-stack",
            shell=True,
            check=True,
        )
