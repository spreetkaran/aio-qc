libmpv - optional playback engine (recommended)
================================================

With libmpv the Frame Extractor plays videos with mpv's engine: hardware decoding on the GPU that drives your
display, display-synchronised timing (59.94 / 60 fps plays evenly), frame-exact seeking. Pausing, stepping, saving
and extraction keep using the Frame Extractor's own exact decoder, and the built-in engine takes over automatically
if libmpv is missing or can't be used.

Automatic: when libmpv is missing the app gets it by itself (Windows / macOS: a background download into your
user folder; Linux: asks once, then installs your distribution's package). Settings > Playback > libmpv... shows
whether it is in use (or exactly why not), runs or turns off the automatic install, and has the manual ways below.

Windows : libmpv-2.dll from an "mpv-dev-lgpl-x86_64" (ARM: "...aarch64...") archive of
          https://github.com/zhongfly/mpv-winbuild/releases  (LGPL-2.1+; the plain mpv-dev builds are GPL).
          Extract only libmpv-2.dll and use "Install DLL..." in that dialog (or put it in this folder).
Linux   : your distribution's package - sudo apt install libmpv2 (Ubuntu 22.04: libmpv1),
          sudo dnf install mpv-libs (RPM Fusion), sudo pacman -S mpv, sudo zypper install libmpv2.
          A libmpv copied from a newer system fails with "undefined symbol": use the package.
macOS   : brew install mpv

App builds can carry it in this folder: python packaging/fetch_libmpv.py (see packaging/LIBMPV.md).
Requirements: the GPU renderer (View > Renderer > Auto or GPU) and mpv 0.33 or newer.
Check it is active: the status bar reads "libmpv · ..."; View > Playback statistics shows the decoder (hwdec),
display sync and dropped frames.
