import glob
import os
from pprint import pprint

import pandas.core.frame
import tensorflow as tf
import PIL
import numpy as np
import uuid

from configuration import Configuration


class TestingUtil:

    @staticmethod
    def test_with_single_image(model, class_names):
        # image_url = f"file:///{os.path.abspath('.')}/test_data/915460/35378_Bright Blue_1_1619488901.jpeg"
        image_url = f"file:///{os.path.abspath('.')}//test_data/2420/P43_56dz_P43_1618572375827.jpg"
        predicted_class, prediction_certainty = TestingUtil.__test_model_with_image(
            model,
            image_url,
            class_names
        )
        print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(
            predicted_class,
            prediction_certainty)
        )

    @staticmethod
    def test_with_test_set(model, class_names, test_data_path=Configuration.test_data_location):
        print(class_names)
        results = []
        for filename in glob.iglob(f"{test_data_path}/**/*.*", recursive=True):
            path = filename.split(os.sep)
            class_name = path[-2]
            predicted_class, prediction_certainty = TestingUtil.__test_model_with_image(
                model,
                f'file:///{filename}',
                class_names,
                display_image=False
            )

            results.append((
                filename.split(os.sep)[-1],
                class_name,
                predicted_class,
                # class_names[predicted_class],
                'Yes' if class_name == predicted_class else 'No'
            ))

        # pprint(results)
        TestingUtil.pretty_print_results(results)
        return results

    @staticmethod
    def pretty_print_results(results):
        print(pandas.core.frame.DataFrame(results, columns=['image', 'class', 'prediction', 'Correct']))
        correct = len(list(filter(lambda x: x[-1] == 'Yes', results)))
        total = len(results)
        print('Test results: {} / {} ({:.2f}%)'.format(correct, total, correct / total * 100))

    @staticmethod
    def __test_model_with_image(model, image_url, class_names, display_image=True):
        # print(f'Testing image: {image_url}')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        brick_path = tf.keras.utils.get_file(str(uuid.uuid4()), origin=image_url)

        img = tf.keras.utils.load_img(
            brick_path, target_size=(Configuration.img_height, Configuration.img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        if display_image:
            display(PIL.Image.open(brick_path))

        # predicted_class = np.argmax(score)
        predicted_class = class_names[np.argmax(score)]
        prediction_certainty = 100 * np.max(score)

        return predicted_class, prediction_certainty
