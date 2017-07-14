# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QAbstractItemView, QLabel, QHeaderView
from PyQt5.QtCore import Qt

import config
import util
import matching
import PyQt5

class ResultsDialog(QDialog):
    def __init__(self, filename, matches, parent=None):
        super(ResultsDialog, self).__init__(parent)
        # Save variables
        self.filename = filename
        self.matches = matches

        # Set-up window
        self.setWindowTitle("Results for " + filename)

        # Set-up layout
        layout = QVBoxLayout()

        # Layout is different depending on the number of matches
        if len(matches) == 0:
            layout.addWidget(QLabel("No matches...."))
        else:
           self.setup_table()
           layout.addWidget(self.table)

        self.setLayout(layout)

    def setup_table(self):
        # Extract the labels
        self.labels = list(self.matches[0].get_results())

        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumWidth(len(self.labels) * 160 + 50)
        self.table.setMinimumHeight(450)
        self.table.setRowCount(len(self.matches))
        self.table.setColumnCount(len(self.labels))
        self.table.setHorizontalHeaderLabels(self.labels)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.doubleClicked.connect(self.on_double_click_table)

        # Sort and show info for the matches
        self.matches = sorted(self.matches, key=lambda x: x.comparison_key(), reverse=True)
        row = 0
        for row in range(len(self.matches)):
            match = self.matches[row]
            values = list(match.get_results().values())

            for i in range(len(values)):
                value = str(values[i])
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, i, item)

    def on_double_click_table(self):
        # Get the rows
        rows = sorted(set(index.row() for index in
                          self.table.selectedIndexes()))

        if (len(rows) > 0):
            # Get the match selected
            row = rows[0]
            match = self.matches[row]

            # Open the experiment results
            for im_info in match.im_results:
                image_diag = ImageDialog(im=im_info.im, title=im_info.title)
                image_diag.show()
                # Save results if needed
                if config.UI_SAVE_RESULTS:
                    filename = str(match.nissl_level) + '-results-' + im_info.filename
                    util.im_write(filename + '.jpg', im_info.im)

class ImageDialog(QDialog):
    def __init__(self, im, title, parent=None):
        super(ImageDialog, self).__init__(parent)

        self.setWindowTitle(title)
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        layout = QVBoxLayout()

        from graph import Graph
        self.canvas = Graph(self, width=20, height=20, dpi=100)
        self.canvas.is_interactive = False
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        self.canvas.imshow(im)

def main():
    app = PyQt5.QtWidgets.QApplication([])
    ui = TestWidget()
    ui.test_result_dialog()
    app.exec()

class TestWidget(PyQt5.QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(TestWidget, self).__init__(parent)
        self.setWindowTitle('Test')

    def test_result_dialog(self):
        import numpy as np
        filename = 'test'
        ransac = {"homography": np.array([[1, 2], [3, 4]]),
                "inlier_mask": np.array([0, 0, 1, 1, 1, 1, 1]),
                "inlier_count": 5,
                "original_inlier_mask": None,
                "total_error": 150.20391,
                'metric': 102.4231,
                "min_error": 0.000000,
                "max_error": 500.203123,
                "avg_error": 15.12314123
                }
        m0 = matching.Match(-5, np.ones(10), ransac, None, np.ones(5), np.ones(5), 100, True)
        matches = [m0, m0, m0, m0]
        diag = ResultsDialog(filename, matches, self)
        diag.show()

if __name__ == '__main__':
    main()