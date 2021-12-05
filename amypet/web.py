from streamlit.bootstrap import load_config_options, run

from . import _backend_web

CONFIG = {
    'browser.gatherUsageStats': False, 'theme.base': 'light', 'theme.primaryColor': '#2a569f',
    'theme.secondaryBackgroundColor': '#a3b5cd', 'theme.textColor': '#000000',
    'theme.font': 'monospace'}


def main():
    load_config_options(CONFIG)
    run(_backend_web.__file__, "", [], CONFIG)


if __name__ == "__main__":
    main()
