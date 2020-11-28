#!/usr/bin/env python3
import logging
import sys
from argparse import SUPPRESS, ArgumentParser, RawDescriptionHelpFormatter
from os import path
from subprocess import PIPE, Popen
from textwrap import dedent

from argopt import argopt
from pkg_resources import resource_filename

from amypad import __version__

try:
    from gooey import Gooey
except ImportError:

    def Gooey(**_):
        def wrapper(func):
            return func

        return wrapper


__licence__ = __license__ = open(
    path.join(path.dirname(path.dirname(__file__)), "LICENCE.md")
).read()
ENCODING = sys.getfilesystemencoding()
log = logging.getLogger(__name__)


class CmdException(Exception):
    def __init__(self, code, cmd, stderr):
        super(CmdException, self).__init__(
            "Code %d:\n===\n%s\n===\n%s" % (code, cmd, stderr)
        )


class Base(object):
    def __init__(self, python_deps=None, matlab_deps=None):
        self.python_deps = python_deps or []
        self.matlab_deps = matlab_deps or []

    def __str__(self):
        return dedent(
            """\
              python_deps:
                - {}
              matlab_deps:
                - {}"""
        ).format("\n  - ".join(self.python_deps), "\n  - ".join(self.matlab_deps))


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
        self.parser = argopt(
            dedent(doc),
            argparser=argparser,
            formatter_class=formatter_class,
            # version=version,
        )
        self.parser.set_defaults(main=self.main)
        self.cmd = cmd
        super(Cmd, self).__init__(**kwargs)

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
                raise CmdException(out.returncode, str(self), stderr.decode(ENCODING))
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
        self.parser = argopt(
            dedent(doc),
            argparser=argparser,
            formatter_class=formatter_class,
            # version=version,
        )
        self.parser.set_defaults(run=func)
        # self.func = func
        super(Func, self).__init__(**kwargs)

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
    cuinfo = Cmd(
        [sys.executable, "-m", "miutil.cuinfo"],
        miutil.cuinfo.__doc__,
        version=miutil.__version__,
        python_deps=["miutil[cuda]"],
        argparser=argparser,
    )
    log.debug(cuinfo)

    # example of how to wrap any callable using an `argopt`-style docstring
    hasext = Func(
        miutil.hasext,
        """\
        Usage:
          hasext <fname> <ext>

        Arguments:
          <fname>  : path to file
          <ext>    : extension (with or without `.` prefix)
        """,
        version=__version__,
        python_deps=["miutil"],
        argparser=argparser,
    )
    log.debug(hasext)

    args = args or sys.argv[1:]
    opts = parser.parse_args(args=args)
    # strip args
    args = [i for i in args if i not in ("--dry-run")]

    if gui_mode:
        print(" ".join([path.basename(sys.executable), "-m amypad"] + args))
    if opts.dry_run:
        pass
    elif hasattr(opts, "main"):  # Cmd
        print(opts.main(args[1:], verify_args=False), end="")
    elif hasattr(opts, "run"):  # Func
        # strip opts
        kwargs = {
            k: v
            for (k, v) in opts._get_kwargs()
            if k not in ("dry_run", "print_command", "run")
        }
        res = opts.run(*opts._get_args(), **kwargs)
        if res is not None:
            print(res)


if __name__ == "__main__":
    main()
