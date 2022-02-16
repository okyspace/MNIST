import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np

tf.enable_v2_behavior()

from clearml import Task
task = Task.init(project_name='mnist_demo', task_name='mnist1')


def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.summary()
    return model


def train(model, epochs, ds_train, ds_test):
    model.fit(ds_train, epochs=epochs, validation_data=ds_test)
    return model


def infer(model, x):
    return model.predict(x)


def save(model, path):
    model.save(path)


def get_info(model):
    model.input
    model.output


def get_data():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )
    return ds_train, ds_test, ds_info


def preprocess(image):
    img = Image.open('/content/7.png').convert('L')
    img = img.resize((28, 28))
    imgArr = np.asarray(img) / 255
    imgArr = np.expand_dims(imgArr[:, :, np.newaxis], 0)
    imgArr = imgArr.astype(np.float32)
    return imgArr


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


if __name__ == '__main__':
    ds_train, ds_test, ds_info = get_data()

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    model = get_model()
    model = train(model, 2, ds_train, ds_test)
    save(model, "model.savedmodel")

    print("Model saved...!!!")
