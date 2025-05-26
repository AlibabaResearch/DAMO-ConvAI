import subprocess
from typing import Literal, Optional
from pydantic import BaseModel
import rich
import rich.errors
from ..rich_pixels import Pixels

from rich.prompt import Prompt
from rich.segment import Segment
from rich.style import Style

from pathlib import Path

import typer
from .menu import Menu
import tempfile

from ..app import app


def _get_system() -> Literal["Linux", "Darwin", "Windows"]:
    import platform

    system = platform.system()
    if system == "Linux":
        return "Linux"
    elif system == "Darwin":
        return "Darwin"
    elif system == "Windows":
        return "Windows"
    else:
        raise ValueError(
            f"You are using {system} which is not supported. Please use Linux, MacOS or Windows."
        )


class Dataset(BaseModel):
    id: str
    display_name: str
    url: str
    venue: str
    license: str
    citation: str


class Datasets(BaseModel):
    datasets: list[Dataset]


def _get_dataset_to_be_loaded(
    published_datasets: Datasets, console: rich.console.Console
) -> str:
    menu = Menu(
        *(
            f"{dataset.display_name} ({dataset.venue}, {dataset.license})"
            for dataset in published_datasets.datasets
        ),
        "None of the above, I want only an empty database.",
        "No, I have a custom URL.",
        start_index=0,
        align="left",
        rule_title="Select the dataset to be loaded",
    )

    dataset = menu.ask(return_index=True)
    assert isinstance(dataset, int)

    if dataset < len(published_datasets.datasets):
        console.log(
            f"""Loading the database with data from {published_datasets.datasets[dataset].url}.
This data is from the {published_datasets.datasets[dataset].display_name}.
Licensed under {published_datasets.datasets[dataset].license}.
Please cite the following paper(s) if you use this data:
{published_datasets.datasets[dataset].citation}"""
        )
        return published_datasets.datasets[dataset].url
    elif dataset == len(published_datasets.datasets):
        console.log("Starting redis with an empty database.")
        return ""
    else:
        custom_load_database_url = Prompt.ask(
            "Enter the URL to load the database with initial data from.",
        )
        if custom_load_database_url == "":
            console.log("Starting redis with an empty database.")
            return ""
        else:
            console.log(
                f"Loading the database with initial data from {custom_load_database_url}."
            )
            return custom_load_database_url


_pixel_art = """\


███████      ███████  ██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
█████        ██       ██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
█████      ███        ████████         ██████████████  ███████████████████████████  ████████████████xx████████████████████████x███████████████xxx█xxx███
█████  ██████         ███████   █████   ███     ███       ████    ████  ██    ██████████     ███████xx██x██xxxxx████xxxxx███xxxxxxx███xxxxx███xxx█xxx███
█████      ███        ████████       ████    ██   ███  █████   ██   ██     █    ██  ██   ██   ██████xx██xxx██xxxx██xx████████xxx████xxx███xx██xxx█xxx███
█████        ██       █████████████     █  █████   ██  ████  █████   █  ██████  ██  ██  ████  ██████xx██xx█████xx██xxxxxx████xxx████xx████xx██xxx█xxx███
█████         ██████  ███████   ██████  █   ████  ███  ████   ████   █   ████   ██  █   ████  ██████xx██xx█████xx████████xx██xxx████xx████xx██xxx█xxx███
█████        ███      █████████        ███       █████    ███      ███  █      ███  ██         █████xx██xx█████xx██xxxxxxxx███xxxxx█xxxxxxxxx█xxx█xxx███
█████       ██        ████████████████████████████████████████████████  ████████████████████████████████████████████████████████████████████████████████
█████   █████       ██████████████████████████████████████████████████  ████████████████████████████████████████████████████████████████████████████████


"""


@app.command()
def install(
    use_docker: Optional[bool] = typer.Option(None, help="Install redis using docker."),
    load_database: Optional[bool] = typer.Option(
        None, help="Load the database with initial data."
    ),
    load_sotopia_pi_data: bool = typer.Option(
        True,
        help="Load the database with initial data from Sotopia π. Only applicable if `load_database` is True.",
    ),
    custom_database_url: Optional[str] = typer.Option(
        None, help="Load the database with initial data from a custom URL."
    ),
    redis_data_path: Optional[str] = typer.Option(
        None,
        help="Path to store the redis data. Only applicable if `use_docker` is True.",
    ),
    overwrite_existing_data: Optional[bool] = typer.Option(
        None, help="Overwrite existing data in the redis data path."
    ),
) -> None:
    console = rich.console.Console()
    mapping = {
        " ": Segment(" ", Style.parse("black on black")),
        "█": Segment("█", Style.parse("white")),
        "x": Segment("x", Style.parse("magenta on magenta")),
    }
    pixels = Pixels.from_ascii(_pixel_art, mapping)
    console.print(pixels, justify="center")
    system = _get_system()

    if use_docker is None:
        if system == "Windows":
            console.log(
                "Installing Redis with Docker... Check if Docker Desktop is installed."
            )
            use_docker = True
        elif system == "Darwin":
            use_docker = (
                Prompt.ask(
                    "Do you want to use Docker or Homebrew to install redis? We recommand you to use Docker.",
                    choices=["Docker", "Homebrew"],
                    default="Docker",
                    console=console,
                )
                == "Docker"
            )
        else:
            use_docker = (
                Prompt.ask(
                    "Do you want to use Docker to install redis or directly install Redis stack binary? We recommand you to use Docker.",
                    choices=["Docker", "Binary"],
                    default="Docker",
                    console=console,
                )
                == "Docker"
            )

    if use_docker:
        try:
            subprocess.check_output("command -v docker", shell=True)
        except subprocess.CalledProcessError:
            if system == "Darwin":
                console.log(
                    """Docker is not installed.
                    Please check https://docs.docker.com/desktop/install/mac-install/,
                    or install it using homebrew with `brew install --cask docker`.
                    And then run this command again.
                    """
                )
            elif system == "Linux":
                console.log(
                    """Docker is not installed.
                    Please check https://docs.docker.com/engine/install/ubuntu/.
                    And then run this command again.
                    """
                )
            else:
                console.log(
                    """Docker is not installed.
                    Please check https://docs.docker.com/desktop/install/windows-install/.
                    And then run this command again.
                    """
                )
            exit(1)
        try:
            subprocess.check_output("docker ps", shell=True)
        except subprocess.CalledProcessError:
            if system == "Darwin":
                console.log(
                    """Docker Daemon was not started. Please launch Docker App.
                    """
                )
            elif system == "Linux":
                console.log(
                    """Docker Daemon was not started. Please run `dockerd`
                    """
                )
            else:
                console.log(
                    """Docker Daemon was not started. Please launch Docker App.
                    """
                )
            exit(1)
    else:
        if system == "Windows":
            console.log("""For Windows, unfortunately only docker is supported.
                Check the official documentation:
                https://redis.io/docs/latest/operate/oss_and_stack/install/install-stack/windows/.
            """)
            exit(1)
        elif system == "Darwin":
            # check if homebrew is installed
            try:
                subprocess.check_output("command -v brew", shell=True)
                subprocess.run("brew update-reset", shell=True, check=True)
            except subprocess.CalledProcessError:
                console.log(
                    """Homebrew is required for install redis without docker on MacOS.
                    Please check https://brew.sh/.
                    And then run this command again.
                    """
                )
                exit(1)

    next_state: Literal["ask_data_source", "empty_database", "custom_url"] | None = None
    url = ""

    if load_database is None:
        load_database = (
            Prompt.ask(
                "Do you want to load the database with published data?",
                choices=["Yes", "No"],
                default="Yes",
                console=console,
            )
            == "Yes"
        )
        if load_database:
            next_state = "ask_data_source"
        else:
            next_state = "empty_database"
    elif not load_database:
        next_state = "empty_database"
    else:
        if load_sotopia_pi_data:
            next_state = "custom_url"
            url = "https://huggingface.co/datasets/cmu-lti/sotopia-pi/resolve/main/dump.rdb?download=true"
        elif custom_database_url is None:
            next_state = "ask_data_source"
        else:
            next_state = "custom_url"
            url = custom_database_url

    if next_state == "ask_data_source":
        fn = Path(__file__).parent / "published_datasets.json"
        published_datasets = Datasets.parse_file(fn)
        url = _get_dataset_to_be_loaded(published_datasets, console)
        next_state = "custom_url"

    assert next_state in ["custom_url", "empty_database"]

    tmpdir_context = tempfile.TemporaryDirectory()
    tmpdir = tmpdir_context.__enter__()

    if url:
        try:
            subprocess.run(f"curl -L {url} -o {Path(tmpdir) / 'dump.rdb'}", shell=True)
            console.log("Database downloaded successfully.")
        except subprocess.CalledProcessError:
            console.log("Database download failed. Please check the URL and try again.")
    else:
        console.log("Starting redis with an empty database.")

    if use_docker:
        current_directory = Path(__file__).parent
        if redis_data_path is None:
            directory = Prompt.ask(
                "Enter the directory where you want to store the data. Press enter to use the current directory.",
                default=current_directory,
            )
        else:
            directory = redis_data_path
        (Path(directory) / "redis-data").mkdir(parents=True, exist_ok=True)
        if load_database:
            if Path.exists(Path(directory) / "redis-data" / "dump.rdb"):
                cover_existing = (
                    Prompt.ask(
                        "The directory already contains a dump.rdb file. Do you want to overwrite it?",
                        choices=["Yes", "No"],
                        default="No",
                        console=console,
                    )
                    if overwrite_existing_data is None
                    else "Yes"
                    if overwrite_existing_data
                    else "No"
                )
                if cover_existing == "No":
                    console.log(
                        "Exiting the installation. Please provide a different directory."
                    )
                    exit(0)
            else:
                (Path(tmpdir) / "dump.rdb").rename(
                    Path(directory) / "redis-data/dump.rdb"
                )
        try:
            subprocess.run(
                f"docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 -v {directory}/redis-data:/data/ redis/redis-stack:latest",
                shell=True,
                check=True,
            )
            console.log(
                "Redis started successfully. To stop redis, run `docker stop redis-stack`. To restart, run `docker restart redis-stack`."
            )
        except subprocess.CalledProcessError:
            console.log("Redis start failed. Please check the logs and try again.")
    else:
        if system == "Darwin":
            try:
                subprocess.run(
                    "brew tap redis-stack/redis-stack", shell=True, check=True
                )
                subprocess.run("brew install redis-stack", shell=True, check=True)
                if load_database:
                    if Path("/opt/homebrew/var/db/redis-stack/dump.rdb").exists():
                        cover_existing = (
                            Prompt.ask(
                                "The directory already contains a dump.rdb file. Do you want to overwrite it?",
                                choices=["Yes", "No"],
                                default="No",
                                console=console,
                            )
                            if overwrite_existing_data is None
                            else "Yes"
                            if overwrite_existing_data
                            else "No"
                        )
                        if cover_existing == "No":
                            console.log(
                                "Exiting the installation. Please provide a different directory."
                            )
                            exit(0)
                    else:
                        Path("/opt/homebrew/var/db/redis-stack/").mkdir(
                            parents=True, exist_ok=True
                        )
                    subprocess.run(
                        f"mv {tmpdir}/dump.rdb /opt/homebrew/var/db/redis-stack/dump.rdb",
                        shell=True,
                        check=True,
                    )
                subprocess.run(
                    "redis-stack-server --daemonize yes", shell=True, check=True
                )
                console.log(
                    "Redis started successfully. To stop redis, run `redis-cli shutdown`. To restart, run this script again."
                )
            except subprocess.CalledProcessError:
                console.log("Redis start failed. Please check the logs and try again.")
        elif system == "Linux":
            try:
                subprocess.run(
                    "curl -fsSL https://packages.redis.io/redis-stack/redis-stack-server-7.2.0-v10.focal.x86_64.tar.gz -o redis-stack-server.tar.gz",
                    shell=True,
                    check=True,
                )
                subprocess.run(
                    "tar -xvzf redis-stack-server.tar.gz", shell=True, check=True
                )
                if load_database:
                    Path("./redis-stack-server-7.2.0-v10/var/db/redis-stack").mkdir(
                        parents=True, exist_ok=True
                    )
                    subprocess.run(
                        f"mv {tmpdir}/dump.rdb ./redis-stack-server-7.2.0-v10/var/db/redis-stack/dump.rdb",
                        shell=True,
                        check=True,
                    )
                subprocess.run(
                    "./redis-stack-server-7.2.0-v10/bin/redis-stack-server --daemonize yes",
                    shell=True,
                    check=True,
                )
                console.log(
                    "Redis started successfully. To stop redis, run `./redis-stack-server-7.2.0-v10/bin/redis-cli shutdown`. To restart, run this script again."
                )
            except subprocess.CalledProcessError:
                console.log("Redis start failed. Please check the logs and try again.")

    tmpdir_context.__exit__(None, None, None)


if __name__ == "__main__":
    app()
