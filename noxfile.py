"""Nox validation sessions for FABRID-IDS.

Sessions run the project environment tools directly (no isolated reinstall),
keeping validation fast and consistent with the active virtual environment.
"""

from __future__ import annotations

import nox

nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ["validation"]

_PYTEST = ".venv/bin/pytest"


def _run(session: nox.Session, *command: str) -> None:
    session.run(*command, external=True)


@nox.session(venv_backend="none")
def format_check(session: nox.Session) -> None:
    _run(session, ".venv/bin/ruff", "format", "--check", ".")


@nox.session(venv_backend="none")
def lint(session: nox.Session) -> None:
    _run(session, ".venv/bin/ruff", "check", ".")


@nox.session(venv_backend="none")
def typecheck(session: nox.Session) -> None:
    _run(session, ".venv/bin/pyright")


@nox.session(venv_backend="none")
def architecture(session: nox.Session) -> None:
    _run(session, _PYTEST, "tests/architecture")


@nox.session(venv_backend="none")
def unit(session: nox.Session) -> None:
    _run(session, _PYTEST, "-m", "not integration and not smoke")


@nox.session(venv_backend="none")
def integration(session: nox.Session) -> None:
    _run(session, _PYTEST, "-m", "integration")


@nox.session(venv_backend="none")
def validation(session: nox.Session) -> None:
    """Complete validation: format, lint, typecheck, and the default test suite."""
    format_check(session)
    lint(session)
    typecheck(session)
    unit(session)


@nox.session(venv_backend="none")
def smoke(session: nox.Session) -> None:
    _run(session, _PYTEST, "-m", "smoke", "-o", "addopts=''")
