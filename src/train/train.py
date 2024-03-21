"""Train neural network model.

Based on
https://www.tensorflow.org/tutorials/images/segmentation
"""

import os
import sys

import dvc.api
import matplotlib.pyplot as plt  # noqa: E402 # fix for GLIBCXX with tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds
from dvclive import Live
from dvclive.keras import DVCLiveCallback
from IPython.display import clear_output
from tensorflow_examples.models.pix2pix import pix2pix

sys.path.append(".")
import helpers as helpers  # noqa: E402

params = dvc.api.params_show()
plot_dir = "eval/training_nn/plots/predictions/"
continue_learning = False
tf.random.set_seed(1)

BATCH_SIZE = params["neural_network"]["batch_size"]
BUFFER_SIZE = params["neural_network"]["buffer_size"]
EPOCHS = params["neural_network"]["epochs"]
VAL_SUBSPLITS = params["neural_network"]["val_subsplit"]
OUTPUT_CLASSES = params["neural_network"]["output_classes"]

dataset, info = tfds.load(params["neural_network"]["dataset"], with_info=True)

if "sim_cp" in params["neural_network"]["dataset"]:
    dataset_type = "coldpools"
    loader = helpers.load_image_crop_n_patch
    display = helpers.raster_display
else:
    dataset_type = "example"
    loader = helpers.load_image_resize
    display = helpers.display

train_images = dataset["train"].map(loader, num_parallel_calls=tf.data.AUTOTUNE).take(1)
print(f"{len(train_images)} train images available")
if loader.__name__ == "load_image_crop_n_patch":
    train_images = train_images.unbatch()  # unbatch requires a data copy
train_images = train_images.take(1)
len_patches_train = len(
    train_images
)  # Note: length of image patches incl. empty masks filtered out next
assert (
    VAL_SUBSPLITS <= len_patches_train
), "Validation subsplits must be smaller than number of images."
train_images = train_images.filter(lambda _, mask: tf.reduce_any(mask != 1))
test_images = (
    dataset["test"].map(loader, num_parallel_calls=tf.data.AUTOTUNE).take(1)
)  # Fix the function call
print(f"{len(test_images)} test images available")
if loader.__name__ == "load_image_crop_n_patch":
    test_images = test_images.unbatch()
len_patches_test = len(test_images)
test_images = test_images.filter(lambda _, mask: tf.reduce_any(mask != 1))


TRAIN_LENGTH = len_patches_train
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
VALIDATION_STEPS = len_patches_train // BATCH_SIZE // VAL_SUBSPLITS

dir_checkpoints = "models/checkpoints/training_checkpoints/"
if not os.path.exists(dir_checkpoints):
    os.makedirs(dir_checkpoints)

assert STEPS_PER_EPOCH > 0, "BATCH_SIZE might be too large. STEPS_PER_EPOCH==0"

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels


train_batches = (
    train_images.cache()
    # .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

test_batches = train_images.batch(BATCH_SIZE)


for images, masks in train_batches.take(1):
    sample_image, sample_mask = images[0], masks[0]

if dataset_type == "coldpools":
    display(
        train_batches.take(100),
        "eval/labels/sample_masks.png",
        "eval/labels/sample_images.png",
    )
elif dataset_type == "example":
    display([sample_image, sample_mask], "eval/labels/sample_images.png")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=[128, 128, 3], include_top=False
)

# Use the activations of these layers
layer_names = [
    "block_1_expand_relu",  # 64x64
    "block_3_expand_relu",  # 32x32
    "block_6_expand_relu",  # 16x16
    "block_13_expand_relu",  # 8x8
    "block_16_project",  # 4x4
]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),  # 32x32 -> 64x64
]


def unet_model(output_channels: int):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips, strict=True):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2, padding="same"
    )  # 64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model = unet_model(output_channels=OUTPUT_CLASSES)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        "binary_accuracy",
        tf.keras.metrics.MeanIoU(
            num_classes=2, sparse_y_true=False, sparse_y_pred=False
        ),
    ],
)

checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(
    checkpoint, directory=dir_checkpoints, max_to_keep=1
)
if continue_learning:
    status = checkpoint.restore(manager.latest_checkpoint)
else:
    # Load checkpoints from https://github.com/Orion-AI-Lab/EfficientBigEarthNet/tree/main
    # wget "https://www.dropbox.com/s/idenhh7g4j3vapb/checkpoint_densenet121.zip?dl=1"
    checkpoint = tf.train.Checkpoint(base_model)
    checkpoint.restore("models/checkpoints/checkpoint_DenseNet121/checkpoints-4.index")


class PredictionsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        helpers.show_predictions(
            model,
            sample_image,
            sample_mask,
            out=f"{plot_dir}/predictions{epoch:04g}.png",
        )


class CreateCheckpoint(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        manager.save(checkpoint_number=1)


with Live("eval/training_nn") as live:
    model_history = model.fit(
        train_batches,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        validation_data=train_batches,  # should potentially be an validation set
        callbacks=[
            PredictionsCallback(),
            DVCLiveCallback(live=live),
            CreateCheckpoint(),
        ],
    )
    test_loss, test_acc, test_iou = model.evaluate(test_batches)
    live.log_metric("test_loss", test_loss, plot=False)
    live.log_metric("test_acc", test_acc, plot=False)
