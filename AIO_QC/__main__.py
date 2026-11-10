# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
import os, argparse, sys, io, contextlib
from loguru import logger

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AIO_QC import __appname__
from AIO_QC import __version__
# Qt and the app are imported in main() after the arguments are read, so `--version` / `--help` work instantly and
# without a display.
QApplication = appIcon = lifecycle = None

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
        from AIO_QC.utils import memory as _mem
        if _mem.is_oom(exc):                                    # out of memory anywhere: free caches, say so once
            logger.opt(exception=(exc_type, exc, tb)).debug("out of memory (traceback)")
            logger.warning(f"[memory] out of memory: {exc_type.__name__}: {str(exc)[:200]} — freeing caches "
                           f"(free memory: {_mem.human(_mem.available())})")
            _mem.release_everywhere()
            app = QApplication.instance()
            img = getattr(getattr(app, "shell", None), "image", None) if app else None
            if img is not None and hasattr(img, "_on_low_memory"):
                try:
                    img._notify_low_memory("last action")
                except Exception:
                    pass
            return
        logger.opt(exception=(exc_type, exc, tb)).error("Unhandled exception")
        app = QApplication.instance()
        win = getattr(app, "main_window", None) if app else None
        if win is not None:
            win.log_errors([f"Unhandled {exc_type.__name__}: {exc}"])
    sys.excepthook = hook


def _process_setup_before_qt():
    """Process-wide pieces that must exist before QApplication — owned here, exactly once for the whole app."""
    import multiprocessing
    multiprocessing.freeze_support()            # Frame Extractor batch workers in frozen (PyInstaller) builds
    if sys.platform == "win32":                 # every child process dies with the app, however it ends
        from AIO_QC.core import procs
        procs.install_job_object()
    try:                                        # vsync + compatibility context for the Frame Extractor's GL canvas
        from PyQt6.QtGui import QSurfaceFormat
        fmt = QSurfaceFormat.defaultFormat()
        fmt.setSwapInterval(1)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
        QSurfaceFormat.setDefaultFormat(fmt)
    except Exception:
        pass


def _process_setup_after_qt(app):
    """After QApplication: C numeric locale (Qt sets the user's on Unix; libmpv refuses to start without "C"), and
    the one app-wide motion filter (dialog fades, smooth wheel scrolling, every window kept on its screen)."""
    if sys.platform != "win32":
        import locale
        try:
            locale.setlocale(locale.LC_NUMERIC, "C")
        except locale.Error:
            pass
    from AIO_QC.widgets import motion
    motion.install(app)


CLI_COMMANDS = ("extract", "probe", "testset", "eval", "clip")
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm", ".wmv", ".flv", ".mpg", ".mpeg", ".ts", ".mts",
              ".m2ts", ".3gp")


def main():
    if len(sys.argv) > 1 and sys.argv[1] in CLI_COMMANDS:
        # the Frame Extractor's command line (Qt-free): aio-qc extract|probe|testset|eval|clip ...
        import multiprocessing
        multiprocessing.freeze_support()
        if sys.platform == "win32":
            from AIO_QC.core import procs
            procs.install_job_object()
        from AIO_QC.tools.frame_extractor.core.cli import main as cli_main
        sys.exit(cli_main(sys.argv[1:]))
    parser = argparse.ArgumentParser(prog="aio-qc", description=f"{__appname__} — quality control for annotated "
                                                                 "image datasets (LabelMe / YOLO)")
    parser.add_argument("--version", "-V", action="store_true", help="show version")
    parser.add_argument(
        "--logger-level",
        default="info",
        choices=["debug", "info", "warning", "fatal", "error"],
        help="logger level",
    )
    parser.add_argument("--mode", choices=["image", "video", "data"], default=None,
                        help="start in the Frame Extractor (video), Image QC or Data Studio (default: Image QC)")
    parser.add_argument("files", nargs="*", help="a video to open in the Frame Extractor, or data files to open in "
                                                 "Data Studio (CSV, Excel, Parquet, JSON …)")
    parser.epilog = ("Frame Extractor command line: aio-qc extract | probe | testset | eval | clip ...  "
                     "(aio-qc extract -h for its options)")
    args = parser.parse_args()

    if args.version:
        print(f"{__appname__} {__version__}")
        sys.exit(0)
        
    global QApplication, appIcon, lifecycle
    from PyQt6.QtWidgets import QApplication
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
    _process_setup_before_qt()
    app = QApplication(sys.argv)
    _process_setup_after_qt(app)
    app.setApplicationName(__appname__)
    app.setApplicationVersion(__version__)
    app.setWindowIcon(appIcon())
    lifecycle.install(app, os.path.dirname(os.path.abspath(LOG_FILE)))   # Qt messages, Ctrl+C, crashes, exit

    _install_excepthook()
    from AIO_QC.shell import AppShell
    shell = AppShell(app)
    app.shell = shell
    app.main_window = shell                  # Ctrl+C / crash hooks: request_quit / log_errors reach both modes
    videos = [f for f in args.files if f.lower().endswith(VIDEO_EXTS)]
    tables = [f for f in args.files if f not in videos]
    mode = "video" if videos else "data" if tables else shell.start_mode(args.mode)
    from AIO_QC.shell import NAMES
    logger.info(f"starting in {NAMES.get(mode, mode)}")

    with contextlib.redirect_stderr(new_target=_LoggerIO()):
        window = shell.show(mode)
        if videos and shell.mode == "video":
            window.load_video(os.path.abspath(videos[0]))
            if len(videos) > 1:                          # several: straight into the batch queue, like a drop
                window.open_batch_queue(add_current=False)
                for v in videos[1:]:
                    window.batch_dialog.add_job(os.path.abspath(v))
        if tables:
            if shell.mode != "data":
                window = shell.window("data")
                shell.switch("data")
            for f in tables:
                window._open_path(os.path.abspath(f))
        code = app.exec()
    logger.debug(f"event loop finished (code {code})")
    return code
        
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()        # packaged builds: process-pool workers (Crop Gallery) start correctly
    main()