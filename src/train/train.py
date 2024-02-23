"""Train neural network model.

Based on
https://www.tensorflow.org/tutorials/images/segmentation
"""

import os
import sys

import dvc.api
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

base_model = tf.keras.applications.DenseNet121(
    include_top=False,
    weights=None,
    input_shape=(128, 128, 3),
    pooling="avg",
)

# Load checkpoints from https://github.com/Orion-AI-Lab/EfficientBigEarthNet/tree/main
# wget "https://www.dropbox.com/s/idenhh7g4j3vapb/checkpoint_densenet121.zip?dl=1"
checkpoint = tf.train.Checkpoint(base_model)
checkpoint.restore("models/checkpoints/checkpoint_DenseNet121/checkpoints-4.index")

# Use the activations of these layers
layer_names = [
    "conv1/relu",  # 60x60 64
    "conv2_block6_concat",  # 30x30 256
    "conv3_block12_concat",  # 15x15 512
    "conv4_block24_concat",  # 7x7 1024
    "conv5_block16_concat",  # 3x3 1024
]

# layer_names = [layer.name for layer in base_model.layers]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False
up_stack = [
    pix2pix.upsample(1024, 3),  # 3x3 -> 7x7
    pix2pix.upsample(512, 3),  # 7x7 -> 15x15
    pix2pix.upsample(256, 3),  # 15x15 -> 30x30
    pix2pix.upsample(128, 3),  # 30x30 -> 64x64
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
    )

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


model = unet_model(output_channels=OUTPUT_CLASSES)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[
        "binary_accuracy",
    ],
)


class PredictionsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        helpers.show_predictions(
            model,
            sample_image,
            sample_mask,
            out=f"{plot_dir}/predictions{epoch:04g}.png",
        )


with Live("eval/training_nn") as live:
    model_history = model.fit(
        train_batches,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        validation_data=test_batches,  # should potentially be an validation set
        callbacks=[PredictionsCallback(), DVCLiveCallback(live=live)],
    )
    test_loss, test_acc = model.evaluate(test_batches)
    live.log_metric("test_loss", test_loss, plot=False)
    live.log_metric("test_acc", test_acc, plot=False)
