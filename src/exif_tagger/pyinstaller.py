import PyInstaller.__main__
from pathlib import Path

HERE = Path(__file__).parent.absolute()
path_to_main = str(HERE / "main.py")


def install():
    PyInstaller.__main__.run(
        [
            path_to_main,
            "--onefile",
            # "--windowed",
            "--collect-all=facenet_pytorch",
            "--add-data=./models/face-recognition.joblib:./models/",
            "--add-data=./db/face-database.sqlite3:./db/",
            "--hidden-import=sklearn.metrics._pairwise_distances_reduction._datasets_pair",
            "--hidden-import=sklearn.metrics._pairwise_distances_reduction._middle_term_computer",
            # "--debug=imports"
            # "--add-data=./db/face-database.sqlite:./db/",
        ]
    )
