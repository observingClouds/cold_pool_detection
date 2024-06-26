import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask, tf.float32) / 255.0
    return input_image, input_mask

def load_image_resize(datapoint):
  """
  Resizing input images to match neural network input layer size.
  """
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(
    datapoint['segmentation_mask'],
    (128, 128),
    method = tf.image.ResizeMethod.NEAREST_NEIGHBOR,
  )

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

def load_image_crop_n_patch(datapoint):
    """
    Crop artificial boundary layer caused by matplotlib figures,
    create patches of specific size matching input layer of model.
    """
    input_image = tf.image.crop_to_bounding_box(
        datapoint["image"],
        offset_height=44,
        offset_width=44,
        target_width=1633,
        target_height=1011,
    )
    image_patches = tf.image.extract_patches(
        images=tf.expand_dims(input_image, axis=0),
        sizes=[1, 128, 128, 1],
        strides=[1, 64, 64, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    image_patches = tf.reshape(image_patches, [-1, 128, 128, 3])
    print(f"{image_patches.shape[0]} patches per image extracted")
    input_mask = tf.image.crop_to_bounding_box(
        datapoint["segmentation_mask"],
        offset_height=44,
        offset_width=44,
        target_width=1633,
        target_height=1011,
    )
    mask_patches = tf.image.extract_patches(
        images=tf.expand_dims(input_mask, axis=0),
        sizes=[1, 128, 128, 1],
        strides=[1, 64, 64, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    mask_patches = tf.reshape(mask_patches, [-1, 128, 128, 1])

    input_image, input_mask = normalize(image_patches, mask_patches)

    return input_image, input_mask


def display(display_list, out=None, _=None):
    plt.figure(figsize=(15, 15))

    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    if out is not None:
        plt.savefig(out)
    else:
        plt.show()


def raster_display(display_list, out_mask=None, out_img=None):
    fig_img, axs_img = plt.subplots(
        int(np.ceil(np.sqrt(len(display_list)))),
        int(np.ceil(np.sqrt(len(display_list)))),
        figsize=(15, 15),
    )
    fig_mask, axs_mask = plt.subplots(
        int(np.ceil(np.sqrt(len(display_list)))),
        int(np.ceil(np.sqrt(len(display_list)))),
        figsize=(15, 15),
    )
    for i, (image, mask) in enumerate(display_list):
        image, mask = image[0], mask[0]
        axs_img.flatten()[i].imshow(tf.keras.utils.array_to_img(image))
        axs_img.flatten()[i].axis("off")
        axs_mask.flatten()[i].imshow(tf.keras.utils.array_to_img(mask))
        axs_mask.flatten()[i].axis("off")
    if out_img is not None:
        if out_img is not None:
            folder_path = os.path.dirname(out_img)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            fig_img.savefig(out_img)
    else:
        fig_img.show()
    if out_mask is not None:
        if out_mask is not None:
            folder_path = os.path.dirname(out_mask)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        fig_mask.savefig(out_mask)
    else:
        fig_mask.show()


def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(model, image, mask, dataset=None, num=1, out=None):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)], out=out)
    else:
        display(
            [
                image,
                mask,
                create_mask(model.predict(image[tf.newaxis, ...])),
            ],
            out=out,
        )
