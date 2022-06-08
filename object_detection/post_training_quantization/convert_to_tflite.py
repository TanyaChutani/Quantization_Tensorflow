import tensorflow as tf


def convert_tflite(tflite_model_path, save_model_path, optimization_method):

  converter = tf.lite.TFLiteConverter.from_saved_model(save_model_path)
  converter.optimizations = [optimization_method]
  tflite_model = converter.convert()

  with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
