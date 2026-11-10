# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
"""Run the recipe on Spark: choose the Python environment (with pyspark), where Spark runs, and the output."""
from __future__ import annotations

import os

from PyQt6.QtCore import QProcess
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
    QVBoxLayout, QWidget,
)

from AIO_QC.core.tabular import spark as SP
from AIO_QC.utils import newButton, style as apply_style

from .step_form import _mono


class SparkDialog(QDialog):
    def __init__(self, parent, ws, table: str, settings):
        super().__init__(parent)
        self.ws, self.table, self.st = ws, table, settings
        self.proc = None
        self.setWindowTitle(f"Run “{table}” on Spark")
        self.resize(760, 560)
        lay = QVBoxLayout(self)
        intro = QLabel("The whole recipe runs on the FULL source files in a Python environment that has pyspark "
                       "(and Java). Local mode uses this computer's cores; a master URL or Spark Connect runs it on "
                       "your cluster. The app itself never loads Spark.")
        intro.setWordWrap(True)
        apply_style(intro, "color:@muted;")
        lay.addWidget(intro)
        form = QFormLayout()
        self.py = QComboBox()
        self.py.setEditable(True)
        try:
            from AIO_QC.core.env_manager import discover_envs
            for name, exe in discover_envs():
                self.py.addItem(f"{name} — {exe}", exe)
        except Exception:
            pass
        last = settings.value("studio/spark_python", "") or ""
        if last:
            i = self.py.findData(last)
            if i < 0:
                self.py.insertItem(0, last, last)
                i = 0
            self.py.setCurrentIndex(i)
        prow = QWidget()
        ph = QHBoxLayout(prow)
        ph.setContentsMargins(0, 0, 0, 0)
        ph.addWidget(self.py, 1)
        ph.addWidget(newButton("Browse…", slot=self._browse))
        ph.addWidget(newButton("Check", slot=self._probe))
        form.addRow("Python environment:", prow)
        self.env_state = QLabel("")
        form.addRow("", self.env_state)
        self.where = QComboBox()
        self.where.addItem("This computer (local[*])", "local")
        self.where.addItem("Cluster master URL (yarn, spark://…, k8s://…)", "master")
        self.where.addItem("Spark Connect (sc://host:15002)", "remote")
        self.url = QLineEdit(settings.value("studio/spark_url", "") or "")
        self.url.setPlaceholderText("yarn · spark://host:7077 · sc://host:15002")
        wrow = QWidget()
        wh = QHBoxLayout(wrow)
        wh.setContentsMargins(0, 0, 0, 0)
        wh.addWidget(self.where)
        wh.addWidget(self.url, 1)
        form.addRow("Run on:", wrow)
        self.out = QLineEdit(settings.value("studio/spark_out", "") or "")
        self.out.setPlaceholderText("Output folder (Spark writes a folder of part files)")
        self.fmt = QComboBox()
        for k in ("parquet", "csv", "json"):
            self.fmt.addItem(k, k)
        orow = QWidget()
        oh = QHBoxLayout(orow)
        oh.setContentsMargins(0, 0, 0, 0)
        oh.addWidget(self.out, 1)
        oh.addWidget(self.fmt)
        oh.addWidget(newButton("…", slot=self._pick_out))
        form.addRow("Write to:", orow)
        lay.addLayout(form)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(_mono())
        lay.addWidget(self.log, 1)
        foot = QHBoxLayout()
        self.save_btn = newButton("Save script…", slot=self._save_script)
        self.run_btn = newButton("Run", icon="run", slot=self._run)
        self.stop_btn = newButton("Stop", slot=self._stop)
        self.stop_btn.setEnabled(False)
        foot.addWidget(self.save_btn)
        foot.addStretch(1)
        foot.addWidget(self.run_btn)
        foot.addWidget(self.stop_btn)
        foot.addWidget(newButton("Close", slot=self.close))
        lay.addLayout(foot)

    def _python(self) -> str:
        return (self.py.currentData() or self.py.currentText()).split(" — ")[-1].strip()

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Python of the environment with pyspark", "",
                                           "Python (python* python*.exe);;All files (*)")
        if p:
            self.py.insertItem(0, p, p)
            self.py.setCurrentIndex(0)

    def _pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "Parent folder for the output")
        if d:
            self.out.setText(os.path.join(d, f"{self.table}_{self.fmt.currentData()}"))

    def _probe(self) -> bool:
        py = self._python()
        if not py:
            self.env_state.setText("Choose a Python first")
            return False
        self.env_state.setText("Checking…")
        self.env_state.repaint()
        try:
            r = SP.probe(py)
        except Exception as exc:
            r = {"pyspark": None, "error": str(exc)}
        if r.get("pyspark"):
            java = "" if r.get("java") else " · ⚠ Java not found (JAVA_HOME / java on PATH) — local Spark needs it"
            self.env_state.setText(f"✓ pyspark {r['pyspark']} · Python {r.get('python', '?')}{java}")
            return True
        self.env_state.setText("✗ no pyspark in this environment — pip install pyspark (and a Java 17 runtime)"
                               + (f"\n{r.get('error', '')[:200]}" if r.get("error") else ""))
        return False

    def _code(self):
        out = self.out.text().strip() or None
        return self.ws.code(self.table, "pyspark", out, self.fmt.currentData())

    def _save_script(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save PySpark script", f"{self.table}_pyspark.py", "Python (*.py)")
        if p:
            with open(p, "w", encoding="utf-8") as f:
                f.write(self._code())

    def _run(self):
        if not self.out.text().strip():
            QMessageBox.information(self, self.windowTitle(), "Choose where Spark should write the result.")
            return
        if not self._probe():
            return
        mode = self.where.currentData()
        if mode != "local" and not self.url.text().strip():
            QMessageBox.information(self, self.windowTitle(), "Type the master URL / Spark Connect address.")
            return
        self.st.setValue("studio/spark_python", self._python())
        self.st.setValue("studio/spark_url", self.url.text().strip())
        self.st.setValue("studio/spark_out", self.out.text().strip())
        script = SP.write_script(self._code())
        cmd = SP.command(self._python(), script, "local[*]" if mode == "local" else
                         (self.url.text() if mode == "master" else ""), self.url.text() if mode == "remote" else "")
        self.log.appendPlainText("$ " + " ".join(cmd) + "\n")
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._done)
        self.proc.start(cmd[0], cmd[1:])
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _read(self):
        if self.proc is not None:
            self.log.appendPlainText(bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace").rstrip())

    def _done(self, code, _status):
        self._read()
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        ok = code == 0
        self.log.appendPlainText(f"\n{'✓ finished — written to ' + self.out.text() if ok else f'✗ exit code {code}'}")
        self.proc = None

    def _stop(self):
        if self.proc is not None:
            self.proc.kill()

    def closeEvent(self, e):
        if self.proc is not None:
            if QMessageBox.question(self, self.windowTitle(), "Spark is still running. Stop it?") != \
                    QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.proc.kill()
            self.proc.waitForFinished(3000)
        super().closeEvent(e)

