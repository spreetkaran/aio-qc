# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: GPL-3.0-only
from PyQt6.QtWidgets import QFileDialog

class VideoLoadDialog(QFileDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFileMode(QFileDialog.FileMode.ExistingFile)
    
    @staticmethod
    def getOpenFileName(parent=None, caption=None, directory="", filter=None):
        if caption is None:
            caption = "Load Video"
        if filter is None:
            from AIO_QC.tools.frame_extractor.core.common import VIDEO_FILTER
            filter = VIDEO_FILTER
            
        dialog = VideoLoadDialog(parent)
        dialog.setWindowTitle(caption)
        dialog.setNameFilter(filter)
        if directory:
            dialog.setDirectory(directory)
        if dialog.exec():
            return dialog.selectedFiles()[0], dialog.selectedNameFilter()
        return '', ''