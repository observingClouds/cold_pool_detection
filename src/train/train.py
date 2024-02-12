"""Train neural network model.

Based on
https://www.tensorflow.org/tutorials/images/segmentation
"""

import sys

import dvc.api
import tensorflow as tf
import tensorflow_datasets as tfds
from dvclive import Live
from dvclive.keras import DVCLiveCallback
from IPython.display import clear_output
from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow_graphics.nn.metric import intersection_over_union

sys.path.append(".")
import helpers as helpers  # noqa: E402

params = dvc.api.params_show()

dataset, info = tfds.load(params["neural_network"]["dataset"], with_info=True)

train_images = dataset["train"].map(
    helpers.load_image, num_parallel_calls=tf.data.AUTOTUNE
)
test_images = dataset["test"].map(
    helpers.load_image, num_parallel_calls=tf.data.AUTOTUNE
)  # Fix the function call

BATCH_SIZE = params["neural_network"]["batch_size"]
BUFFER_SIZE = params["neural_network"]["buffer_size"]
EPOCHS = params["neural_network"]["epochs"]
VAL_SUBSPLITS = params["neural_network"]["val_subsplit"]
OUTPUT_CLASSES = params["neural_network"]["output_classes"]

TRAIN_LENGTH = info.splits["train"].num_examples
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
VALIDATION_STEPS = info.splits["test"].num_examples // BATCH_SIZE // VAL_SUBSPLITS

assert STEPS_PER_EPOCH > 0, "BATCH_SIZE might be too large. STEPS_PER_EPOCH==0"


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
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

test_batches = test_images.batch(BATCH_SIZE)


for images, masks in train_batches.take(2):
    sample_image, sample_mask = images[0], masks[0]
    helpers.display([sample_image, sample_mask])

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


model = unet_model(output_channels=OUTPUT_CLASSES)
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)


helpers.show_predictions(model, sample_image, sample_mask)


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        helpers.show_predictions(model, sample_image, sample_mask)
        print("\nSample Prediction after epoch {}\n".format(epoch + 1))


with Live("custom_dir") as live:
    model_history = model.fit(
        train_batches,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        validation_data=test_batches,  # should potentially be an validation set
        callbacks=[DisplayCallback(), DVCLiveCallback(live=live)],
    )
    test_loss, test_acc = model.evaluate(test_batches)
    live.log_metric("test_loss", test_loss, plot=False)
    live.log_metric("test_acc", test_acc, plot=False)
    live.log_metric("iou", intersection_over_union(test_batches, model), plot=True)
    live.log_plot(
        "prediction", helpers.show_predictions(model, sample_image, sample_mask)
    )
