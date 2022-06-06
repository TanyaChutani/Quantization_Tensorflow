import tensorflow as tf

class TensorflowLiteModel:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self._input_details = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()

    def model_predict(self, image_path):
        image = load_image(image_path)
        self.interpreter.set_tensor(self._input_details[0]["index"], image)
        self.interpreter.invoke()
        tflite_interpreter_output = self.interpreter.get_tensor(self._output_details[0]["index"])
        scores = self.interpreter.get_tensor(self._output_details[0]["index"])
        coordinates = self.interpreter.get_tensor(self._output_details[1]["index"])
        labels = self.interpreter.get_tensor(self._output_details[3]["index"])

        return scores, coordinates, labels

