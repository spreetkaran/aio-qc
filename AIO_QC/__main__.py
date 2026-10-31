# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
import os, argparse, sys, io, contextlib
from loguru import logger

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AIO_QC import __appname__
from AIO_QC import __version__
# Qt and the app are imported in main() after the arguments are read, so `--version` / `--help` work instantly and
# without a display.
QApplication = MainWindow = appIcon = lifecycle = None

class _LoggerIO(io.StringIO):
    def write(self, message: str) -> int:
        if stripped_message := message.strip():
            logger.debug(stripped_message)
        return len(message)

    def flush(self) -> None:
        pass

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return False

from AIO_QC.core.appdirs import log_dir as _log_dir

LOG_FILE = str(_log_dir() / "aio_qc_tool.log")      # source: current folder · exe: <data>/logs (core.appdirs)


def _setup_loguru(logger_level: str) -> None:
    try:
        logger.remove(handler_id=0)
    except ValueError:
        pass

    if sys.stderr:
        logger.add(sys.stderr, level=logger_level)

    log_file = LOG_FILE
    logger.add(
        log_file,
        colorize=False,
        level="DEBUG",
        rotation="10 MB",
        retention="1 year",
        compression="gz",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

def _install_excepthook() -> None:
    """PyQt6 aborts the process on an unhandled exception inside a slot/virtual.
    Log it instead so one bad file never kills a QC session."""
    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):           # Ctrl+C surfacing inside a slot
            lifecycle.set_exit_reason("Ctrl+C from the terminal")
            logger.warning("Ctrl+C — closing AIO QC gracefully")
            app = QApplication.instance()
            win = getattr(app, "main_window", None) if app else None
            if win is not None:
                win.request_quit("Ctrl+C")
            elif app is not None:
                app.quit()
            return
        logger.opt(exception=(exc_type, exc, tb)).error("Unhandled exception")
        app = QApplication.instance()
        win = getattr(app, "main_window", None) if app else None
        if win is not None:
            win.log_errors([f"Unhandled {exc_type.__name__}: {exc}"])
    sys.excepthook = hook


def main():
    parser = argparse.ArgumentParser(prog="aio-qc", description=f"{__appname__} — quality control for annotated "
                                                                 "image datasets (LabelMe / YOLO)")
    parser.add_argument("--version", "-V", action="store_true", help="show version")
    parser.add_argument(
        "--logger-level",
        default="info",
        choices=["debug", "info", "warning", "fatal", "error"],
        help="logger level",
    )
    args = parser.parse_args()

    if args.version:
        print(f"{__appname__} {__version__}")
        sys.exit(0)
        
    global QApplication, MainWindow, appIcon, lifecycle
    from PyQt6.QtWidgets import QApplication
    from AIO_QC.app import MainWindow
    from AIO_QC.utils import appIcon
    from AIO_QC.widgets import lifecycle
    _setup_loguru(logger_level=args.logger_level.upper())
    logger.info(f"Starting {__appname__} {__version__} · Python {sys.version.split()[0]} · {sys.platform}")
    try:
        code = _run(args)
    except KeyboardInterrupt:                       # Ctrl+C before the window existed (or during start-up)
        lifecycle.set_exit_reason("Ctrl+C during start-up")
        logger.warning("Interrupted (Ctrl+C) before AIO QC finished starting")
        code = 130
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 0
    except BaseException:
        lifecycle.set_exit_reason("crash during start-up")
        logger.exception("AIO QC could not start")
        code = 1
    lifecycle.set_exit_code(code)
    sys.exit(code)


def _run(args) -> int:
    
    if sys.platform == "win32":
        # Own taskbar identity: otherwise Windows groups the window under python.exe and shows Python's icon
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(f"AIO_QC.{__version__}")
        except Exception:
            pass
    app = QApplication(sys.argv)
    app.setApplicationName(__appname__)
    app.setApplicationVersion(__version__)
    app.setWindowIcon(appIcon())
    lifecycle.install(app, os.path.dirname(os.path.abspath(LOG_FILE)))   # Qt messages, Ctrl+C, crashes, exit

    _install_excepthook()
    window = MainWindow()
    app.main_window = window

    with contextlib.redirect_stderr(new_target=_LoggerIO()):
        window.show()
        window.raise_()
        code = app.exec()
    logger.debug(f"event loop finished (code {code})")
    return code
        
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()        # packaged builds: process-pool workers (Crop Gallery) start correctly
    main()