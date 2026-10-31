# SPDX-FileCopyrightText: 2026 Preetkaran Singh
# SPDX-License-Identifier: MIT
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QToolBar, QWidgetAction, QToolButton

class ToolBar(QToolBar):
    def __init__(self, title):
        super().__init__(title)
        layout = self.layout()
        m = (0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setContentsMargins(*m)
        self.setContentsMargins(*m)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)

    def addAction(self, action):  # type: ignore[override]
        if isinstance(action, QWidgetAction):
            return super().addAction(action)
        btn = QToolButton()
        btn.setDefaultAction(action)
        btn.setToolButtonStyle(self.toolButtonStyle())
        btn.setIconSize(self.iconSize())          # widgets added to a toolbar do not follow its icon size
        self.addWidget(btn)

        # center align
        for i in range(self.layout().count()):
            if isinstance(self.layout().itemAt(i).widget(), QToolButton):
                self.layout().itemAt(i).setAlignment(Qt.AlignmentFlag.AlignCenter)