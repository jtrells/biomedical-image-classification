""" Predict the modality of a batch of images based on the hierarchical setup """

from typing import Dict, List
from pandas import DataFrame
from numpy import hstack, ndarray
from torch import no_grad, max, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from dataset.ImageDataset import EvalImageDataset
from models.ResNetClass import ResNetClass

# pylint: disable=line-too-long
hierarchy = {
    "classifier": "higher-modality",
    "classname": None,
    "path": "/media/cumulus/curation_data/vil-al-interface/models/cord19/higher-modality/higher-modality_1.pt",
    "classes": ["exp", "gra", "mic", "mol", "oth", "pho", "rad"],
    "children": [
        {
            "classifier": "experimental",
            "classname": "exp",
            "path": "/media/cumulus/curation_data/vil-al-interface/models/cord19/experimental/experimental_1.pt",
            "classes": ["exp.gel", "exp.pla"],
            "children": [],
        },
        {
            "classifier": "graphics",
            "classname": "gra",
            "path": "/media/cumulus/curation_data/vil-al-interface/models/cord19/graphics/graphics_1.pt",
            "classes": [
                "gra.3dr",
                "gra.flow",
                "gra.his",
                "gra.lin",
                "gra.oth",
                "gra.sca",
                "gra.sig",
            ],
            "children": [],
        },
        {
            "classifier": "microscopy",
            "classname": "mic",
            "path": "/media/cumulus/curation_data/vil-al-interface/models/cord19/microscopy/microscopy_1.pt",
            "classes": ["mic.ele", "mic.flu", "mic.lig"],
            "children": [
                {
                    "classifier": "electron",
                    "classname": "mic.ele",
                    "path": "/media/cumulus/curation_data/vil-al-interface/models/cord19/microscopy/electron_1.pt",
                    "classes": ["mic.ele.sca", "mic.ele.tra"],
                    "children": [],
                },
            ],
        },
        {
            "classifier": "molecular",
            "classname": "mol",
            "path": "/media/cumulus/curation_data/vil-al-interface/models/cord19/molecular/molecular_1.pt",
            "classes": ["mol.3ds", "mol.che", "mol.dna", "mol.pro"],
            "children": [],
        },
        {
            "classifier": "radiology",
            "classname": "rad",
            "path": "/media/cumulus/curation_data/vil-al-interface/models/cord19/radiology/radiology_1.pt",
            "classes": ["rad.ang", "rad.cmp", "rad.uls", "rad.oth", "rad.xra"],
            "children": [],
        },
    ],
}


class ImagePredictor:
    """Class responsible for predicting the modalities of a list of images by
    traversing all the hierarchy of classifiers"""

    def __init__(
        self,
        classifiers: Dict,
        device: str = "cuda:0",
        batch_size: int = 128,
        num_workers: int = 1,
    ):
        self.classifiers = classifiers
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _load_dataframe(self, image_names: List[str]) -> DataFrame:
        """Load a dummy dataframe because the Dataset requires the data to
        be organized in a dataframe.
        """
        df_images = DataFrame(columns=["path"], data=image_names)
        df_images["prediction"] = None
        return df_images

    def _load_dataloader(
        self, data: DataFrame, base_path: str, mean: Tensor, std: Tensor
    ) -> DataLoader:
        """Dataloader only containing inputs, no labels"""
        test_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean.numpy(), std.numpy()),
            ]
        )

        test_dataset = EvalImageDataset(
            data,
            base_img_dir=base_path,
            image_transform=test_transform,
            path_col="path",
        )

        dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return dataloader

    def _load_label_encoder(self, prediction_classes: List[str]) -> LabelEncoder:
        encoder = LabelEncoder()
        encoder.fit(prediction_classes)
        return encoder

    def predict(self, model: ResNetClass, dataloader: DataLoader) -> ndarray:
        """Predict the classes for a given model"""
        model.to(self.device)
        model.eval()
        predictions = []
        with no_grad():
            for batch in dataloader:
                data = batch.to(self.device)
                batch_predictions = model(data)
                _, batch_predictions = max(batch_predictions, dim=1)
                batch_predictions = batch_predictions.cpu()
                predictions.append(batch_predictions)
        del model  # free memory
        return hstack(predictions)

    def predict_for_hierarchy(
        self, image_names: List[str], base_path: str
    ) -> DataFrame:
        """Predict the classes of the images in image_names, located at
        base_path, by traversing the hierarchy of classifiers in BFS mode"""
        df_images = self._load_dataframe(image_names)

        # traverse classifier tree in BFS to filter predictions by level
        fringe = [self.classifiers]
        while len(fringe) > 0:
            model_node = fringe.pop(0)
            if model_node["children"]:
                fringe += model_node["children"]

            model = ResNetClass.load_from_checkpoint(model_node["path"])
            mean = model.hparams["mean_dataset"]
            std = model.hparams["std_dataset"]

            filtered_df = df_images[df_images.prediction == model_node["classname"]]
            dataloader = self._load_dataloader(filtered_df, base_path, mean, std)
            encoder = self._load_label_encoder(model_node["classes"])
            predictions = self.predict(model, dataloader)
            filtered_df["prediction"] = encoder.inverse_transform(predictions)

            # merge update dataframes with latest predicted values
            df_images = df_images.set_index("path")
            df_images.update(filtered_df.set_index("path"))
            df_images = df_images.reset_index()

        return df_images
