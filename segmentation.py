"""Module for segmentation model inference.

@author: ZiyiW Now Sup
"""
import os

import numpy as np
import tensorflow as tf
from absl import app, flags
from scipy.ndimage import zoom

from utils import constants, io

# define flags
FLAGS = flags.FLAGS

flags.DEFINE_string("image_type", "vent", "either ute or vent for segmentation")
flags.DEFINE_string("nii_filename", "", "nii image file path")

flags.mark_flag_as_required("nii_filename")


def predict(
    image: np.ndarray, image_type: str = constants.ImageType.VENT.value
) -> np.ndarray:
    """Generate a segmentation mask from the proton or ventilation image.

    Args:
        image: np.nd array of the input image to be segmented.
        image_type: str of the image type ute or vent.
    Returns:
        mask: np.ndarray of type bool of the output mask.
    """
    # Shape of the Image
    img_h, img_w, _ = np.shape(image)

    # Dealing with different image shape
    if img_h == 64 and img_w == 64:
        print("Reshaping image for segmentation")
        image = zoom(abs(image), [2, 2, 2])
    elif img_h == 128 and img_w == 128:
        pass
    else:
        raise ValueError("Segmentation Image size should be 128 x 128 x n")

    # Setting Model Weight Directory
    if image_type == constants.ImageType.VENT.value:

        from models.serialized_batch_vnet_Relu import vnet

        model = vnet(input_size=(128, 128, 128, 1))
        weights_dir_current = "./models/weights/model_ANATOMY_UTE.h5"
    elif image_type == "vent":

        from models.serialized_batch_vnet_Relu_model_batch import vnet

        model = vnet(input_size=(128, 128, 128, 1))
        weights_dir_current = "./models/weights/model_ANATOMY_VEN.h5"
    else:
        raise ValueError("image_type must be ute or vent")

    # Loading model weights, those are trained elsewhere
    model.load_weights(weights_dir_current)

    if image_type == "ute":
        img_thre = np.percentile(image, 99)
        image = np.divide(image, img_thre)
        image[image > 1] = 1
        image = np.multiply(image, 255)

        # this is used for sup 3D CNN
        image = (image - 51.99475164450344) / (48.790268508069744)
        image = image[None, ...]
        image = image[..., None]

    elif image_type == "vent":
        image = np.abs(image)
        image = 255 * (image - np.min(image)) / (np.max(image) - np.min(image))

        ven_mean = np.mean(image)
        ven_std = np.std(image)
        image = (image - ven_mean) / (ven_std)
    else:
        raise ValueError("Image type must be ute or vent")
    # Model Prediction
    image = image[None, ...]
    image = image[..., None]
    mask = model.predict(image)
    # Making mask binary
    mask = mask[0, :, :, :, 0]
    mask[mask > 0.5] = 1
    mask[mask < 1] = 0
    return mask.astype(bool)


def main(argv):
    """Run CNN model inference on ute or vent image."""
    image = io.import_nii(FLAGS.nii_filename)
    image_type = FLAGS.image_type
    mask = predict(image, image_type)
    export_path = os.path.join(os.path.dirname(FLAGS.nii_filename), "mask.nii")
    io.export_nii(image=mask.astype("float64"), path=export_path)


if __name__ == "__main__":
    app.run(main)
