from joblib import dump, load
from pathlib import Path
import numpy as np
from facenet_pytorch import (
    InceptionResnetV1,
    MTCNN,
    extract_face,
    fixed_image_standardization,
)
import torch
import PIL.Image
import json
import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


from exif_tagger.database import Face, UNKNOWN_PERSON, UNKNOWN_PICTURE


MINIMUM_NB_EMBEDDINGS_PER_PERSON: int = 2


class FaceRecognitionModel:
    def __init__(
        self,
        pca_model: PCA,
        gauss_model: GaussianMixture,
        person_ids: np.ndarray,
    ) -> None:
        # Select CUDA device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Running on device: {}".format(self.device))
        # Create MTCNN model, used for detecting faces from a given image
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
        )
        # Create Resnet model, used for generating embeddings from faces
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        # Create PCA model, used for reducing the dimensionality of embbedings
        # (and thus file size)
        self.pca_model = pca_model
        # Create Gaussian mixture model, for classifying known and unknown faces
        self.gauss_model = gauss_model
        # Set list of person ids appearing in the gaussian model
        self.person_ids = person_ids

    def train_person(
        self,
        person_id: int,
        embeddings: np.ndarray,
    ) -> None:
        print("UPDATING PERSON")
        if len(embeddings) < MINIMUM_NB_EMBEDDINGS_PER_PERSON:
            print(
                f"Not enough embeddings for person #{person_id}! Have {len(embeddings)}, need {MINIMUM_NB_EMBEDDINGS_PER_PERSON}."
            )
            return
        # print("####### EMBEDDINGS AFTER #######", embeddings)

        small_embeddings = self.pca_model.transform(embeddings)
        gm = GaussianMixture(n_components=1)
        print(f"Fitting with embeddings of shape {small_embeddings.shape}")
        gm.fit(small_embeddings)

        # Update Gaussian model
        n_components = np.sum(self.person_ids != person_id) + 1
        # print(self.person_ids != person_id)
        # print(f"{self.gauss_model.means_.shape=}")
        # print(f"{self.gauss_model.means_.shape=}")
        # print(f"{self.gauss_model.means_[self.person_ids != person_id].shape=}")
        # print(f"{self.gauss_model.precisions_cholesky_.shape=}")
        # print(f"{gm.precisions_cholesky_.shape=}")
        self.gauss_model.means_ = np.concatenate(
            [self.gauss_model.means_[self.person_ids != person_id], gm.means_]
        )
        self.gauss_model.covariances_ = np.concatenate(
            [
                self.gauss_model.covariances_[self.person_ids != person_id],
                gm.covariances_,
            ]
        )
        self.gauss_model.precisions_cholesky_ = np.concatenate(
            [
                self.gauss_model.precisions_cholesky_[self.person_ids != person_id],
                gm.precisions_cholesky_,
            ]
        )
        w = np.ones(
            (len(self.gauss_model.weights_) - (person_id in self.person_ids) + 1,)
        )
        self.person_ids = np.concatenate(
            [
                self.person_ids[self.person_ids != person_id],
                [person_id],
            ]
        )
        w[self.person_ids != UNKNOWN_PERSON.id] = 1.0
        # w[self.person_ids == UNKNOWN_PERSON.id] = 0
        self.gauss_model.weights_ = w / np.sum(w)
        print(self.gauss_model.means_)

    def infer_person(self, embedding: np.ndarray):
        print("PREDICTING PERSON")
        small_embedding = self.pca_model.transform([embedding])
        predicted_person_id = self.person_ids[
            self.gauss_model.predict(small_embedding)[0]
        ]
        # print(
        #     list(
        #         zip(self.gauss_model.predict_proba(small_embedding)[0], self.person_ids)
        #     )
        # )
        return predicted_person_id

    def from_embeddings(
        embeddings_filename: Path,
        classes_filename: Path,
        target_pca_embedding_size: int = 128,
    ) -> "FaceRecognitionModel":
        embeddings = np.load(embeddings_filename)
        names = np.array(json.load(open(classes_filename)))

        n_components = 0
        means = []
        covariances = []
        weights = []
        precisions_cholesky = []

        pca_model = PCA(n_components=target_pca_embedding_size)
        small_embeddings = pca_model.fit_transform(embeddings)

        for name in tqdm.tqdm(set(names)):
            gm = GaussianMixture(n_components=1)
            gm.fit(small_embeddings[names == name])
            n_components += len(gm.means_)
            means += [gm.means_]
            covariances += [gm.covariances_]
            weights += [gm.weights_]
            precisions_cholesky += [gm.precisions_cholesky_]

        gauss_model = GaussianMixture(n_components=n_components)
        gauss_model.means_ = np.concatenate(means, axis=0)
        gauss_model.covariances_ = np.concatenate(covariances, axis=0)
        gauss_model.precisions_cholesky_ = np.concatenate(precisions_cholesky, axis=0)
        gauss_model.weights_ = np.ones((n_components,)) / np.sum(n_components)

        person_ids = np.array([UNKNOWN_PERSON.id] * n_components)

        return FaceRecognitionModel(
            pca_model=pca_model,
            gauss_model=gauss_model,
            person_ids=person_ids,
        )

    def from_file(model_filename: Path) -> "FaceRecognitionModel":
        return FaceRecognitionModel(**load(open(model_filename, "rb")))

    def to_file(self, model_filename: Path) -> None:
        dump(
            {
                "pca_model": self.pca_model,
                "gauss_model": self.gauss_model,
                "person_ids": self.person_ids,
            },
            open(model_filename, "wb"),
            compress=9,
        )

    def detect_faces(
        self, img: PIL.Image, infer_from_model: bool = False
    ) -> list[Face]:
        print("Detecting faces...")
        boxes, probs = self.mtcnn.detect(img=img)
        print("Detect boxes proba:", probs)

        if boxes is None:
            return []

        print("Extracting faces...")
        faces_tensors = [
            fixed_image_standardization(extract_face(img=img, box=bbox).float())
            for bbox in boxes
        ]

        # print("Generating embeddings...")
        # print(len(faces_tensors), faces_tensors[0].shape)
        aligned = torch.stack(faces_tensors).to(self.device)
        # print(aligned.shape)
        print(faces_tensors[0])
        embeddings: torch.Tensor = self.resnet(aligned).detach().cpu()

        # print("###### EMBEDDING START ######", embeddings)

        faces = [
            Face(
                person_id=(
                    self.infer_person(embedding)
                    if infer_from_model
                    else UNKNOWN_PERSON.id
                ),
                picture_id=UNKNOWN_PICTURE.id,
                confirmed=False,
                bbox=[int(d) for d in bbox],
                embedding=embedding.numpy(),
            )
            for bbox, embedding in zip(boxes, embeddings)
        ]
        return faces
