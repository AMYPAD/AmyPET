from streamlit.bootstrap import run

from . import _backend_web


def main():
    run(_backend_web.__file__, "", [], {})


if __name__ == "__main__":
    main()
