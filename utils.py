import tensorflow as tf
import os
from pathlib import Path
import glob
import shutil
import PIL
import numpy as np


class HelperUtils:

    @staticmethod
    def load_data(url, name):
        data_dir = tf.keras.utils.get_file(
            name,
            origin=f"file:///{os.path.abspath(url)}",
            extract=True
        )
        data_dir = Path(data_dir).parent.as_posix()
        return data_dir

    @staticmethod
    def convert_dataset(data_dir):
        for filename in glob.iglob(f"{data_dir}/**/*.*", recursive=True):
            path = filename.split(os.sep)
            class_name = path[-2]
            new_dir = os.path.join(data_dir, class_name)
            os.makedirs(new_dir, exist_ok=True)
            shutil.move(filename, os.path.join(new_dir, path[-1]))

    @staticmethod
    def test(model, class_names):
        brick_url = f"file:///{os.path.abspath('.')}/test_data/bs20-80nd-bs20-1618908672383.jpg"
        brick_path = tf.keras.utils.get_file('brick.jpg', origin=brick_url)

        img = tf.keras.utils.load_img(
            brick_path, target_size=(Configuration.img_height, Configuration.img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        image = PIL.Image.open(brick_path)
        display(image)

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
                # class_names[np.argmax(score)],
                np.argmax(score),
                100 * np.max(score)
            )
        )


class Configuration:
    batch_size = 32
    img_height = 180
    img_width = 180
    validation_split = 0.2
    epochs = 10
    models_location = './models'
