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
from packaging.version import Version
from streamlit.version import _get_installed_streamlit_version

from amypet.gui import BaseParser, __licence__, __version__, get_main_parser, patch_argument_kwargs

NONE = ''
PARSER = '==PARSER=='
log = logging.getLogger(__name__)
THIS = Path(__file__).parent
CONFIG = {
    'page_title': "AmyPET", 'page_icon': str(THIS / "program_icon.png"), 'layout': 'wide',
    'initial_sidebar_state': 'expanded'}
if _get_installed_streamlit_version() >= Version("0.88.1"):
    CONFIG['menu_items'] = {
        "Get help": "https://github.com/AMYPAD/AmyPET/issues", "Report a Bug": None, "About": f"""
AmyPET Pipeline

***version**: {__version__}

*GUI to run AmyPET tools* ([Source Code](https://github.com/AMYPAD/amypet)).

An https://amypad.eu Project.

{__licence__}"""}

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
    st.set_page_config(**CONFIG)
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
                        left, right = st.columns(2)
                        with left:
                            st.caption(opt.dest)
                        with right:
                            clicked = st.button("Browse directories", **kwargs)
                        key = f'{key_prefix}{opt.dest}_val'
                        if clicked:
                            st.session_state[key] = tk.filedialog.askdirectory(
                                master=root, initialdir=dflt) or dflt
                        val = st.session_state.get(key, dflt)
                        with left:
                            st.write(f"`{val or '(blank)'}`")
                    elif opt.widget == "IntegerField":
                        dflt = opt.default or 0
                        val = st.number_input(opt.dest,
                                              min_value=int(parser.options[opt.dest]['min']),
                                              max_value=int(parser.options[opt.dest]['max']),
                                              value=dflt, **kwargs)
                    elif opt.widget == "DecimalField":
                        dflt = opt.default or 0.0
                        val = st.number_input(opt.dest,
                                              min_value=float(parser.options[opt.dest]['min']),
                                              max_value=float(parser.options[opt.dest]['max']),
                                              format="%g",
                                              step=float(parser.options[opt.dest]['increment']),
                                              value=dflt, **kwargs)
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
                    k = st.sidebar.radio(opt.help,
                                         options=sorted(set(opt.choices) - {'completion'}),
                                         key=f"{key_prefix}{opt.dest}")
                else:
                    k = st.sidebar.radio(opt.dest,
                                         options=sorted(set(opt.choices) - {'completion'}),
                                         **kwargs)
                recurse(opt.choices[k], f"{key_prefix}{k.replace('_', ' ')}_")
            else:
                st.warning(f"Unknown option type:{opt}")

    recurse(parser)
    st.sidebar.image(str(THIS / "config_icon.png"))

    parser = opts.pop(PARSER)
    left, right = st.columns([1, 2])
    with left:
        st.write("**Command**")
    with right:
        prefix = st.checkbox("Prefix")
    cmd = [Path(sys.executable).resolve().name, "-m", parser.prog] + [
        (f"--{k.replace('_', '-')}"
         if v is True else f"--{k.replace('_', '-')}={shlex.quote(str(v))}")
        for k, v in opts.items()]
    st.code(" ".join(cmd if prefix else cmd[2:]), "shell")
    dry_run = not st.button("Run")
    if dry_run:
        log.debug(opts)
    elif 'main__' in parser._defaults: # Cmd
        with st.spinner("Running"):
            st.write(parser._defaults['main__'](cmd[3:], verify_args=False))
    elif 'run__' in parser._defaults:  # Func
        with st.spinner("Running"):
            st.write(parser._defaults['run__'](**opts))
    else:
        st.error("Unknown action")


if __name__ == "__main__":
    main()
