#!/usr/bin/env python3
import logging
import sys
from argparse import SUPPRESS, ArgumentParser, RawDescriptionHelpFormatter
from os import path
from subprocess import PIPE, Popen
from textwrap import dedent
from weakref import WeakSet

from argopt import argopt
from pkg_resources import resource_filename

try:
    from . import __licence__, __version__
except ImportError:
    __version__, __licence__ = "", "Apache-2.0"


try:
    from gooey import Gooey
except ImportError:

    def Gooey(**_):
        def wrapper(func):
            return func

        return wrapper


ENCODING = sys.getfilesystemencoding()
log = logging.getLogger(__name__)


class CmdException(Exception):
    def __init__(self, code, cmd, stdout, stderr):
        super(CmdException, self).__init__(
            dedent(
                """\
                Code {:d}:
                === command ===
                {}
                === stderr ===
                {}=== stdout ===
                {}==="""
            ).format(code, cmd, stderr, stdout)
        )


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

        return dedent(
            """\
            .
              version: {}
              python_deps:{}
              matlab_deps:{}"""
        )[2:].format(self.version, pydeps, mdeps)


class Cmd(Base):
    def __init__(
        self,
        cmd,
        doc,
        version=None,
        argparser=ArgumentParser,
        formatter_class=RawDescriptionHelpFormatter,
        **kwargs
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
            # version=version,
        )
        self.parser.set_defaults(main__=self.main)
        self.cmd = cmd

    def __str__(self):
        return dedent(
            """\
            {}
            {}"""
        ).format(self.parser.prog, super(Cmd, self).__str__())

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
        argparser=ArgumentParser,
        formatter_class=RawDescriptionHelpFormatter,
        **kwargs
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
            # version=version,
        )
        self.parser.set_defaults(run__=func)
        # self.func = func

    def __str__(self):
        return dedent(
            """\
            {}
            {}"""
        ).format(self.parser.prog, super(Func, self).__str__())


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


@Gooey(
    default_size=(768, 768),
    # progress_regex="^\s*(?P<percent>\d[.\d]*)%|",
    # progress_expr="float(percent or 0)",
    # hide_progress_msg=True,
    program_name="amypad",
    sidebar_title="pipeline",
    image_dir=resource_filename(__name__, ""),
    show_restart_button=False,
    # richtext_controls=True,
    header_bg_color="#ffffff",
    sidebar_bg_color="#a3b5cd",
    body_bg_color="#a3b5cd",
    footer_bg_color="#2a569f",
    terminal_font_family="monospace",
    menu=[
        {
            "name": "Help",
            "items": [
                {
                    "type": "Link",
                    "menuTitle": "Source code",
                    "url": "https://github.com/AMYPAD/amypad",
                },
                {
                    "type": "AboutDialog",
                    "menuTitle": "About",
                    "name": "AMYPAD Pipeline",
                    "description": "GUI to run AMYPAD tools",
                    "version": __version__,
                    "copyright": "2020",
                    "website": "https://amypad.eu",
                    "developer": "https://github.com/AMYPAD",
                    "license": __licence__,
                },
            ],
        }
    ],
)
def main(args=None, gui_mode=True):
    logging.basicConfig(level=logging.INFO)
    import miutil.cuinfo

    parser = fix_subparser(
        ArgumentParser(prog=None if gui_mode else "amypad"), gui_mode=gui_mode
    )
    sub_kwargs = {}
    if sys.version_info[:2] >= (3, 7):
        sub_kwargs["required"] = True
    subparsers = parser.add_subparsers(help="pipeline to run", **sub_kwargs)

    def argparser(prog, description=None, epilog=None, formatter_class=None):
        """handle (prog, description, epilog) => (title, help)"""
        return fix_subparser(
            subparsers.add_parser(
                {"miutil.cuinfo": "cuinfo"}.get(prog, prog),  # override
                help="\n".join([description or "", epilog or ""]).strip(),
            ),
            gui_mode=gui_mode,
        )

    # example of how to wrap any CLI command using an `argopt`-style docstring
    Cmd(
        [sys.executable, "-m", "miutil.cuinfo"],
        miutil.cuinfo.__doc__,
        version=miutil.__version__,
        python_deps=["miutil[cuda]"],
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
          <fname>  : path to file
          <ext>    : extension (with or without `.` prefix)
        """,
        version=miutil.__version__,
        python_deps=["miutil"],
        argparser=argparser,
    )

    args = args or sys.argv[1:]
    opts = parser.parse_args(args=args)
    # strip args
    args = [i for i in args if i not in ("--dry-run",)]

    if gui_mode:
        print(" ".join([path.basename(sys.executable), "-m amypad"] + args))
    if opts.dry_run:
        pass
    elif hasattr(opts, "main__"):  # Cmd
        print_not_none(opts.main__(args[1:], verify_args=False), end="")
    elif hasattr(opts, "run__"):  # Func
        # strip opts
        kwargs = {
            k: v for (k, v) in opts._get_kwargs() if k not in ("dry_run", "run__")
        }
        print_not_none(opts.run__(*opts._get_args(), **kwargs))


if __name__ == "__main__":  # pragma: no cover
    main()
