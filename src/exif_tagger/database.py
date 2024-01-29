from pathlib import Path
import sqlite3
import pandas as pd
import json
from dataclasses import dataclass


@dataclass
class Picture:
    filename: Path | str
    id: int | None = None

    def fromSQL(row):
        id, filename = row
        return Picture(filename=Path(filename), id=id)


@dataclass
class Face:
    person_id: int
    picture_id: int
    bbox: tuple[int, int, int, int]
    id: int | None = None

    def fromSQL(row):
        id, person_id, picture_id, x1, y1, x2, y2 = row
        return Face(
            person_id=person_id, picture_id=picture_id, bbox=(x1, y1, x2, y2), id=id
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
            x int,
            y int,
            w int,
            h int,
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
        self.conn = sqlite3.connect(database=database_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.createTable(self.faces_table_creation)
        self.createTable(self.persons_table_creation)
        self.createTable(self.pictures_table_creation)
        if self.getUnknownPerson() is None:
            self.createPerson(Person(name="Unknown", surname="Unknown"))
        self.UNKNOWN_PERSON = self.getUnknownPerson()

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
        cur.execute(sql, (picture.id,))
        self.conn.commit()

    def getPictureById(self, picture_id: int) -> Picture | None:
        sql = "SELECT id, filename FROM pictures WHERE id = ?"
        cur = self.conn.cursor()
        cur.execute(sql, (picture_id,))

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
        cur.execute(sql, (person.name, person.surname))
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
        cur.execute(sql, (person.name, person.surname))
        self.conn.commit()
        return cur.lastrowid

    def deletePerson(self, person):
        sql = "DELETE FROM persons WHERE id=?"
        cur = self.conn.cursor()
        cur.execute(sql, (person.id,))
        self.conn.commit()

    def createFace(self, face: Face):
        sql = """
            INSERT INTO faces(person_id, picture_id, x, y, w, h)
                VALUES(?, ?, ?, ?, ?, ?)
        """
        cur = self.conn.cursor()
        x, y, w, h = face.bbox
        cur.execute(sql, (face.person_id, face.picture_id, x, y, w, h))
        self.conn.commit()
        return cur.lastrowid

    def getPersonById(self, person_id):
        sql = "SELECT id, name, surname FROM persons WHERE id=?"
        cur = self.conn.cursor()
        cur.execute(sql, (person_id,))

        row = cur.fetchone()
        return Person.fromSQL(row)

    def getPersonsByNameSurname(self, name: str, surname: str):
        sql = "SELECT id, name, surname FROM persons WHERE name=? AND surname=?"
        cur = self.conn.cursor()
        cur.execute(sql, (name, surname))

        rows = cur.fetchall()
        return [Person.fromSQL(row) for row in rows]

    def updatePerson(self, person: Person):
        if self.getPersonById(person.id) == self.UNKNOWN_PERSON:
            print("The unknown person cannot be edited!")
            return
        sql = """
            UPDATE persons
            SET name = ?, surname = ?
            WHERE id = ?;
        """
        cur = self.conn.cursor()
        cur.execute(
            sql,
            (person.name, person.surname, person.id),
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
            SET person_id = ?, picture_id = ?, x = ?, y = ?, w = ?, h = ?
            WHERE id = ?;
        """
        cur = self.conn.cursor()
        x, y, w, h = face.bbox
        cur.execute(
            sql,
            (face.person_id, face.picture_id, x, y, w, h, face.id),
        )
        self.conn.commit()

    def getFacesByPersonId(self, person_id: int):
        sql = (
            "SELECT id, person_id, picture_id, x, y, w, h FROM faces WHERE person_id=?"
        )
        cur = self.conn.cursor()
        cur.execute(sql, (person_id,))

        rows = cur.fetchall()
        return [Face.fromSQL(row) for row in rows]

    def getFacesByPictureId(self, picture_id: int):
        sql = (
            "SELECT id, person_id, picture_id, x, y, w, h FROM faces WHERE picture_id=?"
        )
        cur = self.conn.cursor()
        cur.execute(sql, (picture_id,))

        rows = cur.fetchall()
        return [Face.fromSQL(row) for row in rows]

    def deleteFacesByPictureId(self, picture_id: int):
        sql = "DELETE FROM faces WHERE picture_id=?"
        cur = self.conn.cursor()
        cur.execute(sql, (picture_id,))
        self.conn.commit()

    def saveFaces(self, faces: list[Face]):
        for face in faces:
            self.createFace(face=face)

    def getFaces(self):
        sql = "SELECT id, person_id, picture_id, x, y, w, h FROM faces"
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

    def __del__(self):
        self.conn.close()
