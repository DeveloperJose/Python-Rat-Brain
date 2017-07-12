# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QAbstractItemView
from PyQt5.QtCore import Qt

import config
import feature

class ResultsDialog(QDialog):
    def __init__(self, filename, matches, parent=None):
        super(ResultsDialog, self).__init__(parent)

        self.setWindowTitle("Results for " + filename)
        #self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)

        self.filename = filename
        self.matches = matches
        #self.labels = ['Plate', 'Match Count', 'Inlier Count', 'Inlier Ratio', 'H Cond #', 'Det H', 'Hu Dist', 'Convex?', "Total Error", "Avg Error", 'TL Det']
        #self.labels = ['Plate', 'Match Count', 'Inlier Count', 'Inlier Ratio', 'Mag1', 'Mag2', 'Angle', 'Cond']
        #self.labels = ['Plate', 'Match Count', 'Inlier Count', 'Linear', 'I/1000', 'Inl Ratio', 'min(x/y, y/x)', 'abs(sin(angle))', 'Error', 'Avg E', 'MI', 'MA']
        self.labels = ['Plate', 'Match Count', 'Inlier Count', 'Linear', 'I/1000', 'Inl Ratio', 'min(x/y, y/x)', 'abs(sin(angle))', 'C#', 'Det']

        layout = QVBoxLayout()

        self.table = QTableWidget()
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumWidth(len(self.labels) * 160 + 100)
        self.table.setMinimumHeight(450)
        self.table.setRowCount(len(matches))
        self.table.setColumnCount(len(self.labels))
        self.table.setHorizontalHeaderLabels(self.labels)

        self.matches = sorted(matches, key=lambda x: x.comparison_key(), reverse=True)
        row = 0
        for match in self.matches:
            string_repr = match.to_string_array()
            for i in range(len(string_repr)):
                self.table.setItem(row, i, QTableWidgetItem(string_repr[i]))

            row += 1

        self.table.doubleClicked.connect(self.on_double_click_table)

        layout.addWidget(self.table)
        self.setLayout(layout)

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
                    feature.im_write(filename + '.jpg', im_info.im)

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