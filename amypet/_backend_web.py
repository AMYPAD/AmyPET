import logging
import shlex
import sys
import tkinter as tk
from argparse import (
    SUPPRESS,
    _HelpAction,
    _StoreAction,
    _StoreTrueAction,
    _SubParsersAction,
    _VersionAction,
)
from pathlib import Path

import streamlit as st

from amypet.gui import BaseParser, get_main_parser, patch_argument_kwargs

NONE = ''
PARSER = '==PARSER=='
log = logging.getLogger(__name__)

root = tk.Tk()
root.withdraw()
root.wm_attributes('-topmost', 1)


class MyParser(BaseParser):
    def add_argument(self, *args, **kwargs):
        kwargs = patch_argument_kwargs(kwargs, gooey=True)
        widget = kwargs.pop('widget', None)
        log.debug("%r, %r", args, kwargs)
        res = super(MyParser, self).add_argument(*args, **kwargs)
        if widget is not None:
            res.widget = widget
        return res


def main():
    logging.basicConfig(level=logging.DEBUG)
    parser = get_main_parser(gui_mode=False, argparser=MyParser)
    opts = {}

    def recurse(parser, key_prefix=""):
        opts[PARSER] = parser
        st.write(f"{'#' * (key_prefix.count('_') + 1)} {parser.prog}")

        for opt in parser._actions:
            if isinstance(opt, (_HelpAction, _VersionAction)) or opt.dest in {'dry_run'}:
                continue
            elif isinstance(opt, _StoreTrueAction):
                val = st.checkbox(opt.dest, value=opt.default, help=opt.help,
                                  key=f"{key_prefix}{opt.dest}")
                if val != opt.default:
                    opts[opt.dest] = val
            elif isinstance(opt, _StoreAction):
                dflt = NONE if opt.default is None else opt.default
                kwargs = {'help': opt.help, 'key': f"{key_prefix}{opt.dest}"}
                if hasattr(opt, 'widget'):
                    if opt.widget == "MultiFileChooser":
                        val = [
                            i.name for i in st.file_uploader(opt.dest, accept_multiple_files=True,
                                                             **kwargs)]
                    elif opt.widget == "FileChooser":
                        val = getattr(
                            st.file_uploader(opt.dest, accept_multiple_files=False, **kwargs),
                            'name', NONE)
                    elif opt.widget == "DirChooser":
                        # https://github.com/streamlit/streamlit/issues/1019
                        val = st.text_input(opt.dest, value=dflt, **kwargs)
                    elif opt.widget == "IntegerField":
                        dflt = opt.default or 0
                        val = int(st.text_input(opt.dest, value=dflt, **kwargs))
                    elif opt.widget == "DecimalField":
                        dflt = opt.default or 0.0
                        val = float(st.text_input(opt.dest, value=dflt, **kwargs))
                    else:
                        st.error(f"Unknown: {opt.widget}")
                        val = dflt
                elif opt.choices:
                    choices = list(opt.choices)
                    val = st.selectbox(opt.dest, index=choices.index(dflt), options=choices,
                                       **kwargs)
                else:
                    val = st.text_input(opt.dest, value=dflt, **kwargs)
                if val != dflt:
                    opts[opt.dest] = val
            elif isinstance(opt, _SubParsersAction):
                if opt.dest == SUPPRESS:
                    k = st.radio(opt.help, options=list(set(opt.choices) - {'completion'}),
                                 key=f"{key_prefix}{opt.dest}")
                else:
                    k = st.radio(opt.dest, options=list(set(opt.choices) - {'completion'}),
                                 **kwargs)
                recurse(opt.choices[k], f"{key_prefix}{k.replace('_', ' ')}_")
            else:
                st.write(opt)

    recurse(parser)
    parser = opts.pop(PARSER)
    st.write("**Command**")
    cmd = [Path(sys.executable).name, f"-m {parser.prog}"] + [
        (f"--{k.replace('_', '-')}"
         if v is True else f"--{k.replace('_', '-')}={shlex.quote(str(v))}")
        for k, v in opts.items()]
    st.write(" ".join(cmd))
    dry_run = not st.button("Run")
    if dry_run:
        log.debug(opts)
    elif 'main__' in parser._defaults: # Cmd
        st.write(parser._defaults['main__'](cmd[2:], verify_args=False))
    elif 'run__' in parser._defaults:  # Func
        st.write(parser._defaults['run__'](**opts))
    else:
        st.error("Unknown action")


if __name__ == "__main__":
    main()
