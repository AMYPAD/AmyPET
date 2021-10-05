#!/usr/bin/env python3
import logging
import re
import sys
from argparse import SUPPRESS, ArgumentParser, RawDescriptionHelpFormatter
from functools import partial
from pathlib import Path
from subprocess import PIPE, Popen
from textwrap import dedent
from weakref import WeakSet

import shtab
from argopt import argopt
from pkg_resources import resource_filename

try:
    from . import __licence__, __version__
except ImportError:
    __version__, __licence__ = "", "MPL-2.0"

log = logging.getLogger(__name__)
WIDGETS = (
    "FileChooser",
    "MultiFileChooser",
    "DirChooser",
    "FileSaver",
    "MultiFileSaver",
    "Slider",
)
RE_DEFAULT = re.compile(f"\\[default: (None:.*?|{'|'.join(WIDGETS)})\\]", flags=re.M)
RE_PRECOLON = re.compile(r"^\s*:\s*", flags=re.M)
ENCODING = sys.getfilesystemencoding()


def patch_argument_kwargs(kwargs, gooey=True):
    kwargs = kwargs.copy()
    if "help" in kwargs:
        kwargs["help"] = RE_PRECOLON.sub("", RE_DEFAULT.sub("", kwargs["help"]))

    default = kwargs.get("default", None)
    if default in WIDGETS:
        if gooey:
            kwargs["widget"] = default
        kwargs["default"] = None
    elif gooey:
        typ = kwargs.get("type", None)
        if typ == open:
            nargs = kwargs.get("nargs", 1)
            if nargs and (nargs > 1 if isinstance(nargs, int) else nargs in "+*"):
                kwargs['widget'] = "MultiFileChooser"
            else:
                kwargs['widget'] = "FileChooser"
        elif typ == int:
            kwargs['widget'] = "IntegerField"
            kwargs['gooey_options'] = {'min': 0, 'max': 1_000}
        elif typ == float:
            kwargs['widget'] = "DecimalField"
            kwargs['gooey_options'] = {'min': -1, 'max': 1e3, 'increment': 1e-6}

    return kwargs


try:
    from gooey import Gooey
    from gooey import GooeyParser as BaseParser

    patch_argument_kwargs = partial(patch_argument_kwargs, gooey=True)
except ImportError:
    BaseParser = ArgumentParser
    patch_argument_kwargs = partial(patch_argument_kwargs, gooey=False)

    def Gooey(**_):
        def wrapper(func):
            return func

        return wrapper


class MyParser(BaseParser):
    def add_argument(self, *args, **kwargs):
        kwargs = patch_argument_kwargs(kwargs)
        log.debug("%r, %r", args, kwargs)
        return super(MyParser, self).add_argument(*args, **kwargs)


class CmdException(Exception):
    def __init__(self, code, cmd, stdout, stderr):
        super(CmdException, self).__init__(
            dedent("""\
                Code {:d}:
                === command ===
                {}
                === stderr ===
                {}=== stdout ===
                {}===""").format(code, cmd, stderr, stdout))


class Base(object):
    _instances = WeakSet()

    def __init__(self, python_deps=None, matlab_deps=None, version=__version__):
        self.python_deps = python_deps or []
        self.matlab_deps = matlab_deps or []
        self.version = version

    def __new__(cls, *_, **__):
        self = object.__new__(cls)
        cls._instances.add(self)
        return self

    def __str__(self):
        pydeps = ""
        if self.python_deps:
            pydeps = "\n  - " + "\n  - ".join(self.python_deps)
        mdeps = ""
        if self.matlab_deps:
            mdeps = "\n  - " + "\n  - ".join(self.matlab_deps)

        return dedent("""\
            .
              version: {}
              python_deps:{}
              matlab_deps:{}""")[2:].format(self.version, pydeps, mdeps)


class Cmd(Base):
    def __init__(
        self,
        cmd,
        doc,
        version=None,
        argparser=MyParser,
        formatter_class=RawDescriptionHelpFormatter,
        **kwargs,
    ):
        """
        Args:
          cmd (list):  e.g. `[sys.executable, "-m", "miutil.cuinfo"]`
          doc (str): an `argopt`-compatible docstring for `cmd`
          version (str): optional
        """
        super(Cmd, self).__init__(**kwargs)
        self.parser = argopt(
            dedent(doc),
            argparser=argparser,
            formatter_class=formatter_class,
            version=version,
        )
        self.parser.set_defaults(main__=self.main)
        self.cmd = cmd

    def __str__(self):
        return dedent("""\
            {}
            {}""").format(self.parser.prog,
                          super(Cmd, self).__str__())

    def main(self, args, verify_args=True):
        """
        Args:
            args (list): list of arguments (e.g. `sys.argv[1:]`)
            verify_args (bool): whether to parse args to ensure no input errors
        """
        try:
            if verify_args:
                self.parser.parse_args(args=args)
        except SystemExit as exc:
            if exc.code:
                raise
        else:
            # return check_output(self.cmd + args, stderr=STDOUT).decode("U8")
            out = Popen(self.cmd + args, stdout=PIPE, stderr=PIPE)
            stdout, stderr = out.communicate()
            if out.returncode != 0:
                raise CmdException(
                    out.returncode,
                    str(self),
                    stdout.decode(ENCODING),
                    stderr.decode(ENCODING),
                )
            return stdout.decode(ENCODING)


class Func(Base):
    def __init__(
        self,
        func,
        doc,
        version=None,
        argparser=MyParser,
        formatter_class=RawDescriptionHelpFormatter,
        **kwargs,
    ):
        """
        Args:
          func (callable):  e.g. `miutil.hasext`
          doc (str): an `argopt`-compatible docstring for `func`
          version (str): optional
        """
        super(Func, self).__init__(**kwargs)
        self.parser = argopt(
            dedent(doc),
            argparser=argparser,
            formatter_class=formatter_class,
            version=version,
        )
        self.parser.set_defaults(run__=func)
        # self.func = func

    def __str__(self):
        return dedent("""\
            {}
            {}""").format(self.parser.prog,
                          super(Func, self).__str__())


def fix_subparser(subparser, gui_mode=True):
    subparser.add_argument(
        "--dry-run",
        action="store_true",
        help="don't run command (implies print_command)" if gui_mode else SUPPRESS,
    )
    return subparser


def print_not_none(value, **kwargs):
    if value is not None:
        print(value, **kwargs)


# progress_regex="^\s*(?P<percent>\d[.\d]*)%|",
# progress_expr="float(percent or 0)",
# hide_progress_msg=True,
# richtext_controls=True,
@Gooey(default_size=(768, 768), program_name="amypad", sidebar_title="pipeline",
       image_dir=resource_filename(__name__, ""), show_restart_button=False,
       header_bg_color="#ffffff", sidebar_bg_color="#a3b5cd", body_bg_color="#a3b5cd",
       footer_bg_color="#2a569f", terminal_font_family="monospace", menu=[{
           "name": "Help", "items": [{
               "type": "Link", "menuTitle": "🌐 View source (online)",
               "url": "https://github.com/AMYPAD/amypad"}, {
                   "type": "AboutDialog", "menuTitle": "🔍 About", "name": "AMYPAD Pipeline",
                   "description": "GUI to run AMYPAD tools", "version": __version__,
                   "copyright": "2021", "website": "https://amypad.eu",
                   "developer": "https://github.com/AMYPAD", "license": __licence__}]}])
def main(args=None, gui_mode=True):
    logging.basicConfig(level=logging.INFO)
    import miutil.cuinfo
    import niftypad.api
    import niftypad.models

    parser = fix_subparser(MyParser(prog=None if gui_mode else "amypad"), gui_mode=gui_mode)
    sub_kwargs = {}
    if sys.version_info[:2] >= (3, 7):
        sub_kwargs["required"] = True
    subparsers = parser.add_subparsers(help="pipeline to run", **sub_kwargs)
    if not gui_mode:
        subparser = subparsers.add_parser("completion", help="Print tab completion scripts")
        shtab.add_argument_to(subparser, "shell", parent=parser)

    def argparser(prog, description=None, epilog=None, formatter_class=None):
        """handle (prog, description, epilog) => (title, help)"""
        return fix_subparser(
            subparsers.add_parser(
                {"miutil.cuinfo": "cuinfo"}.get(prog, prog),               # override
                help="\n".join([description or "", epilog or ""]).strip(),
            ),
            gui_mode=gui_mode,
        )

    kinetic_model = Func(
        niftypad.api.kinetic_model,
        """\
        Kinetic modelling

        Usage:
          kinetic_model [options] <src>

        Arguments:
          <src>  : Input file/folder [default: FileChooser]

        Options:
          -d PATH, --dst PATH      : Output file/folder (default: input folder)
                                     [default: FileChooser]
          -m MODEL, --model MODEL  : model [default: srtmb_basis]
          -p FILE, --params FILE   : config file hint (relative to `--input`)
                                     [default: FileChooser]
          --w W                : (default: None)
          --r1 R1              : [default: 0.905:float]
          --k2p K2P            : [default: 0.000250:float]
          --beta_lim BETA_LIM  : (default: None)
          --n_beta N_BETA      : [default: 40:int]
          --linear_phase_start LINEAR_PHASE_START  : [default: 500:int]
          --linear_phase_end LINEAR_PHASE_END      : (default: None)
          --km_outputs KM_OUTPUTS                  : (default: None)
          --thr THR                                : [default: 0.1:float]
        """,
        version=niftypad.__version__,
        python_deps=["niftypad>=1.1.0"],
        argparser=argparser,
    )
    opts = kinetic_model.parser._get_optional_actions()
    model = next(i for i in opts if i.dest == "model")
    model.choices = niftypad.models.NAMES

    # example of how to wrap any CLI command using an `argopt`-style docstring
    Cmd(
        [sys.executable, "-m", "miutil.cuinfo"],
        miutil.cuinfo.__doc__,
        version=miutil.__version__,
        python_deps=["miutil[cuda]>=0.8.0"],
        argparser=argparser,
    )

    # example of how to wrap any callable using an `argopt`-style docstring
    Func(
        miutil.hasext,
        """\
        Check if a given filename has a given extension

        Usage:
          hasext <fname> <ext>

        Arguments:
          <fname>  : path to file [default: FileChooser]
          <ext>    : extension (with or without `.` prefix)
        """,
        version=miutil.__version__,
        python_deps=["miutil>=0.8.0"],
        argparser=argparser,
    )

    if args is None:
        args = sys.argv[1:]
    args = [i for i in args if i not in ("--ignore-gooey",)]
    opts = parser.parse_args(args=args)
    args = [i for i in args if i not in ("--dry-run",)] # strip args

    if gui_mode:
        print(" ".join([Path(sys.executable).name, "-m amypad"] + args))
    if opts.dry_run:
        pass
    elif hasattr(opts, "main__"):                                                    # Cmd
        print_not_none(opts.main__(args[1:], verify_args=False), end="")
    elif hasattr(opts, "run__"):                                                     # Func
        kwargs = {k: v
                  for (k, v) in opts._get_kwargs() if k not in ("dry_run", "run__")} # strip opts
        print_not_none(opts.run__(*opts._get_args(), **kwargs))


if __name__ == "__main__": # pragma: no cover
    main()
