#!/usr/bin/env python3
import sys

if "--ignore-gooey" not in sys.argv:
    sys.argv.insert(1, "--ignore-gooey")  # must be before gooey import
from .gui import main

main(gui_mode=False)
