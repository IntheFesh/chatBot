"""``python -m twin`` is the same as the ``twin`` command."""

from twin.cli import app

if __name__ == "__main__":
    app(prog_name="twin")
