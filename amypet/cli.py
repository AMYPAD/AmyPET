#!/usr/bin/env python3
import sys
from functools import partial

if "--ignore-gooey" not in sys.argv:
    sys.argv.insert(1, "--ignore-gooey") # must be before gooey import
from .gui import main as gui_main

main = partial(gui_main, gui_mode=False)

if __name__ == "__main__": # pragma: no cover
    main()
