import io
import sys
from PyQt6.QtCore import *
from PyQt6.QtCore import QObject, Qt
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtWebEngineWidgets import QWebEngineView
from pathlib import Path
from datetime import datetime
from PyQt6.QtWidgets import QWidget
import folium
import json
import numpy as np
import PIL.Image
import PIL.ExifTags
from shapely.geometry import Polygon
from shapely import to_geojson
from facenet_pytorch import MTCNN
from exif_tagger.database import Face, Person, FaceDatabase, UNKNOWN_PERSON


class LineEditDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)

    def createEditor(
        self, parent: QWidget, option: QStyleOptionViewItem, index: QModelIndex
    ):
        editor = QLineEdit(parent)
        editor.setFrame(False)
        return editor

    def setEditorData(self, editor: QLineEdit, index: QModelIndex):
        value = index.model().data(index, Qt.ItemDataRole.EditRole)
        editor.setText(value)

    def setModelData(
        self, editor: QLineEdit, model: QAbstractItemModel, index: QModelIndex
    ):
        model.setData(index, editor.text(), Qt.ItemDataRole.EditRole)

    def updateEditorGeometry(
        self, editor: QLineEdit, option: QStyleOptionViewItem, index: QModelIndex
    ):
        editor.setGeometry(option.rect)


class PersonListModel(QAbstractTableModel):
    def __init__(self, parent: QObject | None, database: FaceDatabase) -> None:
        super().__init__(parent)
        self.database = database
        self.headers = ["id", "name", "surname"]
        self.persons = [
            person for person in self.database.getPersons() if person != UNKNOWN_PERSON
        ]

    def reset(self):
        self.beginResetModel()
        self.persons = [
            person for person in self.database.getPersons() if person != UNKNOWN_PERSON
        ]
        self.endResetModel()

    def deleteIndexes(self, indexes: list[QModelIndex]):
        for row in sorted([index.row() for index in indexes], reverse=True):
            self.database.deletePerson(self.persons[row])
            self.beginRemoveRows(QModelIndex(), row, row)
            self.removeRow(row)
            del self.persons[row]
            self.endRemoveRows()

    def rowCount(self, parent):
        # How many rows are there?
        return len(self.persons)

    def columnCount(self, parent):
        # How many columns?
        return len(self.headers)

    def data(self, index, role):
        if role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return QVariant()
        # What's the value of the cell at the given index?
        return self.persons[index.row()].__getattribute__(self.headers[index.column()])

    def setData(self, index: QModelIndex, value: QVariant, role):
        person = self.persons[index.row()]
        person.__setattr__(self.headers[index.column()], value)
        self.database.updatePerson(person)
        self.dataChanged.emit(index, index)

    def headerData(self, section, orientation, role):
        if (
            role != Qt.ItemDataRole.DisplayRole
            or orientation != Qt.Orientation.Horizontal
        ):
            return QVariant()
        # What's the header for the given column?
        return self.headers[section]

    def flags(self, index: QModelIndex):
        if index.column() == 0:
            return super().flags(index)
        return super().flags(index) | Qt.ItemFlag.ItemIsEditable


class PersonDatabaseDialog(QDialog):
    def __init__(self, parent: QWidget | None, database: FaceDatabase) -> None:
        super().__init__(parent)
        self.setWindowTitle("Person Database")
        self.setModal(True)
        self.database = database

        self.model = PersonListModel(self, database)
        self.view = QTableView(self)
        self.view.setItemDelegate(LineEditDelegate())
        self.view.setModel(self.model)
        self.view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.view.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

        self.addButton = QPushButton("Add Person")
        self.addButton.clicked.connect(self.addPerson)
        self.deleteSelectedButton = QPushButton("Delete Selected")
        self.deleteSelectedButton.clicked.connect(self.deleteSelected)

        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.deleteSelectedButton)
        self.button_layout.addWidget(self.addButton)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.view)
        self.layout.addLayout(self.button_layout)
        self.setLayout(self.layout)

        self.setFixedSize(800, 800)

    def addPerson(self):
        add_person_dialog = AddPersonDialog(self.database)
        add_person_dialog.exec()
        if add_person_dialog.person is not None:
            self.database.createPerson(add_person_dialog.person)
            self.model.reset()

    def deleteSelected(self):
        self.model.deleteIndexes(self.view.selectionModel().selectedRows())


class AddPersonDialog(QDialog):
    def __init__(self, database: FaceDatabase, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add a new Person")
        self.setModal(True)
        self.setMinimumWidth(500)

        self.database = database
        self.person = None

        self.error_label = QLabel()
        self.error_label.setHidden(True)
        self.error_label.setStyleSheet(
            "QLabel { background-color : red; color : white; }"
        )

        self.nameLineEdit = QLineEdit()
        self.nameLineEdit.textEdited.connect(self.validate)
        self.surnameLineEdit = QLineEdit()
        self.surnameLineEdit.textEdited.connect(self.validate)
        self.doneButton = QPushButton("Add")
        self.doneButton.clicked.connect(self.done)
        self.cancelButton = QPushButton("Cancel")
        self.doneButton.clicked.connect(self.reject)

        self.form_layout = QFormLayout()
        self.form_layout.addRow("name", self.nameLineEdit)
        self.form_layout.addRow("surname", self.surnameLineEdit)

        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.cancelButton)
        self.button_layout.addWidget(self.doneButton)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.error_label)
        self.layout.addLayout(self.form_layout)
        self.layout.addLayout(self.button_layout)
        self.setLayout(self.layout)

        self.validate()

    def validate(self, *args):
        self.person = None
        self.doneButton.setDisabled(True)
        self.error_label.setHidden(True)

        name = self.nameLineEdit.text()
        surname = self.surnameLineEdit.text()
        if name == "" or surname == "":
            return False
        name = self.nameLineEdit.text()
        surname = self.surnameLineEdit.text()
        person = Person(name=name, surname=surname)
        if self.database.hasPerson(person):
            self.error_label.setText(f"Person {person.present()} already exists!")
            self.error_label.setHidden(False)
            return False

        self.doneButton.setDisabled(False)
        self.person = person
        return True
