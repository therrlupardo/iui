import glob
import os
import shutil
from pathlib import Path
import tensorflow as tf
from configuration import Configuration
import json


class HelperUtils:

    @staticmethod
    def load_data(url, name):
        file_url = f"file:///{os.path.abspath(url)}"
        print(f'Loading data from {file_url}',)
        data_dir = tf.keras.utils.get_file(
            name,
            origin=file_url,
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
    def save_model(model, class_names):
        print(f'Saving model {Configuration.model_name} and it\'s {len(class_names)} classes')
        model.save(f'{Configuration.models_location}/{Configuration.model_name}.h5')

        with open(f"{Configuration.models_location}/{Configuration.model_name}_class_names.json", "w") as file:
            json.dump(class_names, file)

    @staticmethod
    def load_model(model_name):
        model = tf.keras.models.load_model(f'{Configuration.models_location}/{model_name}.h5')

        with open(f"{Configuration.models_location}/{model_name}_class_names.json") as file:
            class_names = json.load(file)

        return model, class_names
