from pathlib import Path
import sqlite3
import pandas as pd
import json
from dataclasses import dataclass
import numpy as np
import io
import PIL.Image
from facenet_pytorch import MTCNN, extract_face


@dataclass
class Picture:
    filename: Path | str | None
    id: int | None = None

    def fromSQL(row):
        id, filename = row
        return Picture(filename=Path(filename), id=id)


@dataclass
class Face:
    person_id: int
    picture_id: int
    confirmed: bool
    bbox: tuple[int, int, int, int]
    embedding: np.ndarray
    id: int | None = None

    def fromSQL(row):
        id, person_id, picture_id, confirmed, x1, y1, x2, y2, embedding = row
        return Face(
            person_id=person_id,
            picture_id=picture_id,
            confirmed=confirmed,
            bbox=(x1, y1, x2, y2),
            embedding=convert_array(embedding),
            id=id,
        )


@dataclass
class Person:
    name: str
    surname: str
    id: int | None = None

    def fromSQL(row):
        id, name, surname = row
        return Person(id=id, name=name, surname=surname)

    def present(self) -> str:
        if self.name == "Unknown" and self.surname == "Unknown":
            return "Unknown"
        return f"{self.name} {self.surname}"

    def __eq__(self, other: "Person"):
        return self.name == other.name and self.surname == other.surname


UNKNOWN_PERSON: Person = Person(id=1, name="Unknown", surname="Unknown")
UNKNOWN_PICTURE: Picture = Picture(id=1, filename=None)


compressor = "zlib"  # zlib, bz2


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr, allow_pickle=True)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out, allow_pickle=True)


class FaceDatabase:
    pictures_table_creation = """
        CREATE TABLE IF NOT EXISTS pictures (
            id integer PRIMARY KEY,
            filename text NOT NULL
        );
    """

    faces_table_creation = """
        CREATE TABLE IF NOT EXISTS faces (
            id integer PRIMARY KEY,
            person_id integer NOT NULL,
            picture_id integer NOT NULL,
            confirmed int,
            x integer,
            y integer,
            w integer,
            h integer,
            embedding blob,
            CONSTRAINT fk_persons
                FOREIGN KEY (person_id)
                REFERENCES persons(id)
                ON DELETE CASCADE
            CONSTRAINT fk_pictures
                FOREIGN KEY (picture_id)
                REFERENCES pictures(id)
                ON DELETE CASCADE
        );
    """
    persons_table_creation = """
        CREATE TABLE IF NOT EXISTS persons (
            id integer PRIMARY KEY,
            name text,
            surname text
        );
    """

    def __init__(self, database_path: Path) -> None:
        database_path.parent.mkdir(exist_ok=True, parents=True)
        self.conn = sqlite3.connect(database=database_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.createTable(self.faces_table_creation)
        self.createTable(self.persons_table_creation)
        self.createTable(self.pictures_table_creation)
        if self.getUnknownPerson() is None:
            print("Creating Unknown Person")
            self.createPerson(Person(name="Unknown", surname="Unknown"))
            assert UNKNOWN_PERSON == self.getUnknownPerson()

    def createTable(self, create_table_sql):
        c = self.conn.cursor()
        c.execute(create_table_sql)

    def createPicture(self, picture: Picture):
        if self.hasPicture(picture):
            print(f"Picture {picture} already exists!")
            return
        sql = """
            INSERT INTO pictures(filename)
                VALUES(?)
        """
        cur = self.conn.cursor()
        cur.execute(sql, (str(picture.filename),))
        self.conn.commit()
        return cur.lastrowid

    def deletePicture(self, picture: Picture):
        sql = "DELETE FROM pictures WHERE id=?"
        cur = self.conn.cursor()
        cur.execute(sql, (int(picture.id),))
        self.conn.commit()

    def getPictureById(self, picture_id: int) -> Picture | None:
        sql = "SELECT id, filename FROM pictures WHERE id = ?"
        cur = self.conn.cursor()
        cur.execute(sql, (int(picture_id),))

        row = cur.fetchone()
        if row is None:
            return None
        return Picture.fromSQL(row)

    def getPictures(self) -> list[Picture]:
        sql = "SELECT id, filename FROM pictures"
        cur = self.conn.cursor()
        cur.execute(sql)

        rows = cur.fetchall()
        return [Picture.fromSQL(row) for row in rows]

    def hasPicture(self, picture: Picture) -> bool:
        sql = """
            SELECT COUNT(*) FROM pictures WHERE filename=?
        """
        cur = self.conn.cursor()
        cur.execute(sql, (str(picture.filename),))
        self.conn.commit()
        return cur.fetchone()[0] > 0

    def getPictureByFilename(self, filename: Path):
        sql = "SELECT id, filename FROM pictures WHERE filename=?"
        cur = self.conn.cursor()
        cur.execute(sql, (str(filename),))

        row = cur.fetchone()
        return Picture.fromSQL(row)

    def hasPerson(self, person: Person) -> bool:
        sql = """
            SELECT COUNT(*) FROM persons WHERE name=? AND surname=?
        """
        cur = self.conn.cursor()
        cur.execute(sql, (str(person.name), str(person.surname)))
        self.conn.commit()
        return cur.fetchone()[0] > 0

    def createPerson(self, person: Person):
        if self.hasPerson(person):
            print(f"Person {person} already exists!")
            return
        sql = """
            INSERT INTO persons(name, surname)
                VALUES(?,?)
        """
        cur = self.conn.cursor()
        cur.execute(sql, (str(person.name), str(person.surname)))
        self.conn.commit()
        return cur.lastrowid

    def deletePerson(self, person):
        sql = "DELETE FROM persons WHERE id=?"
        cur = self.conn.cursor()
        cur.execute(sql, (int(person.id),))
        self.conn.commit()

    def createFace(self, face: Face):
        sql = """
            INSERT INTO faces(person_id, picture_id, confirmed, x, y, w, h, embedding)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        """
        cur = self.conn.cursor()
        x, y, w, h = face.bbox
        assert (
            self.getPictureById(face.picture_id) is not None
        ), f"No such picture as {face.picture_id=}"
        assert (
            self.getPersonById(face.person_id) is not None
        ), f"No such person as {face.person_id=}"

        cur.execute(
            sql,
            (
                int(face.person_id),
                int(face.picture_id),
                int(face.confirmed),
                x,
                y,
                w,
                h,
                adapt_array(face.embedding),
            ),
        )

        self.conn.commit()
        return cur.lastrowid

    def getPersonById(self, person_id):
        sql = "SELECT id, name, surname FROM persons WHERE id=?"
        cur = self.conn.cursor()
        cur.execute(sql, (int(person_id),))

        row = cur.fetchone()
        return Person.fromSQL(row)

    def getEmbeddingsForPerson(self, person_id):
        sql = "SELECT embedding FROM faces WHERE person_id=?"
        cur = self.conn.cursor()
        cur.execute(sql, (int(person_id),))

        rows = cur.fetchall()
        embeddings = [convert_array(embedding) for (embedding,) in rows]
        return embeddings

    def getPersonsByNameSurname(self, name: str, surname: str):
        sql = "SELECT id, name, surname FROM persons WHERE name=? AND surname=?"
        cur = self.conn.cursor()
        cur.execute(sql, (str(name), str(surname)))

        rows = cur.fetchall()
        return [Person.fromSQL(row) for row in rows]

    def updatePerson(self, person: Person):
        assert (
            self.getPersonById(person.id) != UNKNOWN_PERSON
        ), "The unknown person cannot be edited!"
        sql = """
            UPDATE persons
            SET name = ?, surname = ?
            WHERE id = ?;
        """
        cur = self.conn.cursor()
        cur.execute(
            sql,
            (str(person.name), str(person.surname), int(person.id)),
        )
        self.conn.commit()

    def getPersons(self) -> list[Person]:
        sql = "SELECT id, name, surname FROM persons"
        cur = self.conn.cursor()
        cur.execute(sql)

        rows = cur.fetchall()
        return [Person.fromSQL(row) for row in rows]

    def updateFace(self, face: Face):
        sql = """
            UPDATE faces
            SET person_id = ?, picture_id = ?, confirmed = ?, x = ?, y = ?, w = ?, h = ?, embedding = ?
            WHERE id = ?;
        """
        cur = self.conn.cursor()
        x, y, w, h = face.bbox
        cur.execute(
            sql,
            (
                int(face.person_id),
                int(face.picture_id),
                int(face.confirmed),
                x,
                y,
                w,
                h,
                adapt_array(face.embedding),
                int(face.id),
            ),
        )
        self.conn.commit()

    def getFacesByPersonId(self, person_id: int, confirmed_only: bool = True):
        sql = """
        SELECT id, person_id, picture_id, confirmed, x, y, w, h, embedding FROM faces
            WHERE person_id=? AND confirmed = ?
        """
        cur = self.conn.cursor()
        cur.execute(sql, (int(person_id), int(confirmed_only)))

        rows = cur.fetchall()
        return [Face.fromSQL(row) for row in rows]

    def getFacesByPictureId(self, picture_id: int):
        sql = "SELECT id, person_id, picture_id, confirmed, x, y, w, h, embedding FROM faces WHERE picture_id=?"
        cur = self.conn.cursor()
        cur.execute(sql, (int(picture_id),))

        rows = cur.fetchall()
        return [Face.fromSQL(row) for row in rows]

    def deleteFacesByPictureId(self, picture_id: int):
        sql = "DELETE FROM faces WHERE picture_id=?"
        cur = self.conn.cursor()
        cur.execute(sql, (int(picture_id),))
        self.conn.commit()

    def saveFaces(self, faces: list[Face]):
        for face in faces:
            self.createFace(face=face)

    def getFaces(self):
        sql = "SELECT id, person_id, picture_id, confirmed, x, y, w, h, embedding FROM faces"
        cur = self.conn.cursor()
        cur.execute(sql)

        rows = cur.fetchall()
        return [Face.fromSQL(row) for row in rows]

    def getUnknownPerson(self):
        sql = "SELECT id, name, surname FROM persons WHERE name=? AND surname=?"
        cur = self.conn.cursor()
        cur.execute(sql, ("Unknown", "Unknown"))

        row = cur.fetchone()
        if row is None:
            return None
        return Person.fromSQL(row)

    def getImagesForPerson(
        self, person_id: int, confirmed_only: bool = True
    ) -> list[PIL.Image.Image]:
        mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
        )
        faces = self.getFacesByPersonId(person_id, confirmed_only=confirmed_only)
        imgs = [
            PIL.Image.fromarray(
                extract_face(
                    img=PIL.Image.open(self.getPictureById(face.picture_id).filename),
                    box=face.bbox,
                )
                .numpy()
                .astype(np.uint8)
                .transpose(1, 2, 0)
            )
            for face in faces
        ]
        return imgs

    def purgeOrphanPictures(self, blank=False):
        pictures = self.getPictures()
        for picture in pictures:
            if not Path(picture.filename).exists():
                print(f"{picture.filename} does not exists!")
                self.deletePicture(picture)

    def __del__(self):
        self.conn.close()
