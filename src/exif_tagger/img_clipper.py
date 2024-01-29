from exif_tagger.database import FaceDatabase, Face, Picture
import PIL.Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
import numpy as np


def get_img(database: FaceDatabase, face: Face):
    picture = database.getPictureById(face.picture_id)

    img = PIL.Image.open(str(picture.filename))
    return img.crop(face.bbox)


def infer(database: FaceDatabase, face: Face):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device: {}".format(device))

    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    mtcnn = MTCNN(
        image_size=160,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
    )

    picture = database.getPictureById(face.picture_id)
    img = PIL.Image.open(picture.filename)

    print(face.bbox)
    aligned = mtcnn.extract(
        img=img,
        batch_boxes=np.array(
            [
                face.bbox,
            ]
        ),
        save_path=None,
    )

    aligned = torch.stack((aligned,)).to(device)
    embeddings = resnet(aligned).detach().cpu()

    return embeddings
