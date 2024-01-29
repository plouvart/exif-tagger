import io
import sys
from PyQt6.QtCore import *
from PyQt6.QtCore import QObject, Qt
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtWebEngineWidgets import QWebEngineView
from pathlib import Path
from datetime import datetime
import folium
import json
import numpy as np
import PIL.Image
import PIL.ExifTags
from shapely.geometry import Polygon
from shapely import to_geojson
from facenet_pytorch import MTCNN
from PyQt6.QtWidgets import QWidget

from exif_tagger.database import Face, Person, Picture, FaceDatabase
from exif_tagger.person_ui import PersonDatabaseDialog


# DEFAULT_IMG_PATH = Path("res/default-img.jpg")
DEFAULT_IMG_PATH = Path(
    "/home/pierre/repositories/exif-tagger/res/PXL_20231207_094042805.jpg"
)

DEFAULT_DB_PATH = Path("/home/pierre/repositories/exif-tagger/db/face-database.sqlite")


lat_sign = {
    "S": -1,
    "N": 1,
}
lon_sign = {
    "W": -1,
    "E": 1,
}


def extract_datetime(metadata: dict) -> datetime:
    datetime_str = metadata.get(PIL.ExifTags.Base.DateTime, None)
    if datetime_str is None:
        return None
    return datetime.strptime(datetime_str, "%Y:%m:%d %H:%M:%S")


def extract_latitude(metadata: dict) -> float | None:
    lat_ref = metadata.get(PIL.ExifTags.GPS.GPSLatitudeRef, None)
    lat = metadata.get(PIL.ExifTags.GPS.GPSLatitude, None)
    if lat is None or lat_ref is None:
        return None
    lat = lat_sign[lat_ref] * np.sum(lat * np.array([1.0, 1 / 60, 1 / 3600]))
    return lat


def extract_longitude(metadata: dict) -> float | None:
    lon_ref = metadata.get(PIL.ExifTags.GPS.GPSLongitudeRef, None)
    lon = metadata.get(PIL.ExifTags.GPS.GPSLongitude, None)
    if lon is None or lon_ref is None:
        return None
    lon = lon_sign[lon_ref] * np.sum(lon * np.array([1.0, 1 / 60, 1 / 3600]))
    return lon


def extract_latlon(metadata: dict) -> tuple[float, float]:
    lat, lon = extract_latitude(metadata), extract_longitude(metadata)
    if lat is None or lon is None:
        return None
    return lat, lon


class MetadataModel(QAbstractTableModel):
    def __init__(self, parent: QObject | None, metadata: dict) -> None:
        super().__init__(parent)
        self.headers = ["tag", "value"]
        self.metadata = []

    def setMetadata(self, exif: dict):
        self.layoutAboutToBeChanged.emit()
        self.metadata = [
            (PIL.ExifTags.TAGS[k], str(v))
            for k, v in exif.items()
            if k in PIL.ExifTags.TAGS
        ]
        self.layoutChanged.emit()

    def rowCount(self, parent):
        # How many rows are there?
        return len(self.metadata)

    def columnCount(self, parent):
        # How many columns?
        return len(self.headers)

    def data(self, index, role):
        if role != Qt.ItemDataRole.DisplayRole:
            return QVariant()
        # What's the value of the cell at the given index?
        return self.metadata[index.row()][index.column()]

    def headerData(self, section, orientation, role):
        if (
            role != Qt.ItemDataRole.DisplayRole
            or orientation != Qt.Orientation.Horizontal
        ):
            return QVariant()
        # What's the header for the given column?
        return self.headers[section]


class ExifEditor(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(400)

        self.metadata: dict = {}
        self.img = None
        self.filename = None

        # Map View
        self.web = QWebEngineView()
        self.web.setFixedSize(400, 300)

        # Metadata List
        self.model = MetadataModel(self, self.metadata)
        self.view = QTableView(self)
        self.view.setModel(self.model)
        self.view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        # Comment Edit
        self.comment = QTextEdit()
        self.comment.textChanged.connect(self.saveMetadata)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.web)
        self.layout.addWidget(self.view)
        self.layout.addWidget(self.comment)
        self.setLayout(self.layout)

    def updateMap(self, ifd):
        location = extract_latlon(ifd)
        angle = ifd.get(PIL.ExifTags.GPS.GPSImgDirection, None)

        if location is not None:
            m = folium.Map(location=location, zoom_start=13)
            folium.Marker(location=location).add_to(m)
            if angle is not None:
                angles = np.array(
                    [x * np.pi / 180 for x in np.linspace(angle + 45, angle - 45, 10)]
                )
                points = np.vstack(
                    [
                        location,
                        [location]
                        + 0.01 * np.vstack([np.cos(angles), np.sin(angles)]).T,
                    ],
                )[:, ::-1]
                folium.GeoJson(
                    data=to_geojson(Polygon(points)),
                ).add_to(m)
            data = io.BytesIO()
            m.save(data, close_file=False)
            self.web.setHtml(data.getvalue().decode())
        else:
            self.web.setHtml(None)

    def updateView(self, exif):
        self.model.setMetadata(exif)
        self.view.setModel(self.model)
        # self.view.adjustSize()

    def loadMetadata(self, filename: Path):
        self.ﬁlename = filename
        self.img = PIL.Image.open(filename)
        self.img_exif = self.img.getexif()
        self.img_ifd = self.img_exif.get_ifd(PIL.ExifTags.IFD.GPSInfo)

        self.updateView(self.img_exif)
        self.updateMap(self.img_ifd)
        self.comment.setText(self.img_exif.get(PIL.ExifTags.Base.UserComment, None))

    def saveMetadata(self):
        if self.img is None or self.filename is None:
            return
        self.img_exif[PIL.ExifTags.Base.UserComment] = self.comment.toPlainText()
        self.img.save(self.filename, exif=self.img_exif)
        self.updateView(self.img_exif)


class PictureArea(QWidget):
    def __init__(
        self,
        database: FaceDatabase,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.rects: list[QGraphicsRectItem] = []
        self.comboboxes: list[QGraphicsProxyWidget] = []

        self.picture = None
        self.faces: list[Face] = []

        self.scene = QGraphicsScene()
        self.view = QGraphicsView()
        self.view.setScene(self.scene)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setBackgroundBrush(Qt.GlobalColor.blue)
        self.view_transform = None

        self.untrackButton = QPushButton("Untrack Picture")
        # self.loadButton = QPushButton("Reload Faces")
        self.detectButton = QPushButton("Detect Faces")
        # self.loadButton.clicked.connect(lambda _: self.loadFaces())
        self.detectButton.clicked.connect(lambda _: self.detectFaces())
        self.untrackButton.clicked.connect(lambda _: self.untrackPicture())

        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.untrackButton)
        # self.button_layout.addWidget(self.loadButton)
        self.button_layout.addWidget(self.detectButton)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.view)
        self.layout.addLayout(self.button_layout)
        self.setLayout(self.layout)

        self.database = database

        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
        )

    def loadImage(self, filename: Path):
        self.img = PIL.Image.open(filename)
        self.picture = Picture(filename=filename)

        self.scene.clear()
        self.rects.clear()
        self.comboboxes.clear()

        pixmap = QPixmap(str(filename), format="jpg")
        if pixmap.width() > pixmap.height():
            r = self.view.width() / pixmap.width()
            self.view_transform = QTransform().scale(r, r)
        else:
            r = self.view.height() / pixmap.height()
            self.view_transform = QTransform().scale(r, r)
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

        self.loadFaces()

    def loadFaces(self):
        if self.database.hasPicture(self.picture):
            self.picture = self.database.getPictureByFilename(self.picture.filename)
            self.faces = self.database.getFacesByPictureId(picture_id=self.picture.id)
        else:
            self.faces = []
        self.displayFaces()

    def trackPicture(self):
        if self.picture is None:
            return
        self.database.createPicture(picture=self.picture)
        self.picture = self.database.getPictureByFilename(
            filename=self.picture.filename
        )

    def detectFaces(self):
        if self.picture is None:
            return
        self.trackPicture()
        self.database.deleteFacesByPictureId(self.picture.id)
        boxes, _ = self.mtcnn.detect(img=self.img)
        boxes = [] if boxes is None else boxes
        self.faces = [
            Face(
                person_id=self.database.UNKNOWN_PERSON.id,
                picture_id=self.picture.id,
                bbox=[int(d) for d in bbox],
            )
            for bbox in boxes
        ]
        for face in self.faces:
            self.database.createFace(face=face)

        self.displayFaces()

    def displayFaces(self):
        all_persons = self.database.getPersons()

        for rect in self.rects:
            self.scene.removeItem(rect)
        for proxy in self.comboboxes:
            self.scene.removeItem(proxy)
        self.rects.clear()
        self.comboboxes.clear()

        for i, face in enumerate(self.faces):
            x1, y1, x2, y2 = face.bbox

            color = (
                Qt.GlobalColor.red
                if face.person_id == self.database.UNKNOWN_PERSON.id
                else Qt.GlobalColor.green
            )
            pen = QPen(color)
            pen.setWidth(8)
            rect = self.scene.addRect(x1, y1, x2 - x1, y2 - y1, pen=pen)
            self.rects.append(rect)

            combobox = QComboBox()
            proxy = self.scene.addWidget(combobox)
            self.comboboxes.append(proxy)
            for person in all_persons:
                combobox.addItem(person.present(), person)
            cur_index = [person.id for person in all_persons].index(face.person_id)
            combobox.setCurrentIndex(cur_index)
            combobox.currentIndexChanged.connect(
                lambda _, ind=i: self.onChangeCombobox(ind)
            )
            FACTOR = 7
            scale = max(self.img.width, self.img.height) / combobox.width() / FACTOR
            proxy.setTransform(QTransform().scale(scale, scale))
            proxy.setPos(
                (x1 + x2) / 2 - combobox.width() * scale * 0.5,
                y1 - combobox.height() * scale * 1.5,
            )

        self.view.setTransform(self.view_transform)

    def onChangeCombobox(self, ind: int):
        combobox = self.comboboxes[ind]
        person: Person = combobox.widget().currentData()
        color = (
            Qt.GlobalColor.red
            if person == self.database.UNKNOWN_PERSON
            else Qt.GlobalColor.green
        )
        face = self.faces[ind]
        face.person_id = person.id
        self.database.updateFace(face)
        pen = QPen(color)
        pen.setWidth(8)
        self.rects[ind].setPen(pen)

    def untrackPicture(self):
        self.database.deletePicture(self.picture)
        self.loadFaces()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)


class MainWidget(QWidget):
    def __init__(self, database: FaceDatabase, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EXIF Editor")
        # self.resize(800, 800)
        self.cur_filename: Path | None = DEFAULT_IMG_PATH

        self.picture_area = PictureArea(database)
        self.exif_editor = ExifEditor()

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.picture_area)
        self.layout.addWidget(self.exif_editor)
        self.setLayout(self.layout)

        self.loadImg(self.cur_filename)

    def loadImg(self, filename: Path):
        print(f"Loading {filename}...")
        self.cur_filename = Path(filename)
        self.picture_area.loadImage(self.cur_filename)
        self.exif_editor.loadMetadata(filename=self.cur_filename)
        self.exif_editor.adjustSize()

    def nextImg(self):
        if self.cur_filename is None:
            return
        folder = self.cur_filename.parent
        files = sorted([f for f in folder.glob("*") if f.suffix in (".jpg", ".jpeg")])
        ind = files.index(self.cur_filename)
        next_ind = (ind + 1) % len(files)
        self.loadImg(Path(files[next_ind]))

    def previousImg(self):
        if self.cur_filename is None:
            return
        folder = self.cur_filename.parent
        files = sorted([f for f in folder.glob("*") if f.suffix in (".jpg", ".jpeg")])
        ind = files.index(self.cur_filename)
        next_ind = (ind - 1) % len(files)
        self.loadImg(Path(files[next_ind]))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Right:
            self.nextImg()
        if event.key() == Qt.Key.Key_Left:
            self.previousImg()


class MainWindow(QMainWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Picture Tagger")
        self.database = FaceDatabase(database_path=DEFAULT_DB_PATH)
        self.main_widget = MainWidget(self.database)

        self.menuBar = QMenuBar(self)

        self.openFileAction = QAction("Open", self)
        self.openFileAction.triggered.connect(self.openFile)
        fileMenu = QMenu("&File", self)
        fileMenu.addAction(self.openFileAction)
        self.menuBar.addMenu(fileMenu)

        self.openPersonAction = QAction("Open", self)
        self.openPersonAction.triggered.connect(self.openPersonDatabase)
        personMenu = QMenu("&Persons", self)
        personMenu.addAction(self.openPersonAction)
        self.menuBar.addMenu(personMenu)

        self.setMenuBar(self.menuBar)

        self.setCentralWidget(self.main_widget)

    def openFile(self):
        filename = QFileDialog.getOpenFileName(
            self, "Open File", "/home", "Images (*.tiff *.jpg)"
        )
        self.main_widget.loadImg(filename[0])

    def openPersonDatabase(self):
        person_dialog = PersonDatabaseDialog(parent=None, database=self.database)
        person_dialog.exec()
        self.main_widget.picture_area.loadFaces()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
