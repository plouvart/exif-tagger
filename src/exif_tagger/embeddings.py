from exif_tagger.database import FaceDatabase, Face, Picture
import PIL.Image
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.utils.data as data_utils
import tqdm


def generate_embeddings(
    image_folder: Path,
) -> np.ndarray:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on device: {}".format(device))

    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    def collate_fn(x):
        return iter(x)

    dataset = datasets.ImageFolder(image_folder)
    classes = {i: c for c, i in dataset.class_to_idx.items()}
    # dataset = data_utils.Subset(dataset, torch.arange(0, len(dataset), 500))
    dataset.idx_to_class = classes
    loader = DataLoader(dataset, batch_size=100, collate_fn=collate_fn, num_workers=2)

    mtcnn = MTCNN(
        image_size=160,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
    )

    all_embeddings = []
    all_classes = []
    for batch in tqdm.tqdm(loader):
        aligned = []
        names = []
        for x, y in batch:
            x_aligned, prob = mtcnn(x, return_prob=True)
            if x_aligned is not None:
                aligned.append(x_aligned)
                names.append(dataset.idx_to_class[y])

        aligned = torch.stack(aligned).to(device)
        embeddings = resnet(aligned).detach().cpu()

        all_embeddings.append(embeddings)
        all_classes.extend(names)

    return torch.concat(all_embeddings, dim=0), all_classes
