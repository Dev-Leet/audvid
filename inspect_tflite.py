import tensorflow as tf
import sys

def inspect(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    output_details = interpreter.get_output_details()

    print(f"Number of outputs: {len(output_details)}")
    for i, output in enumerate(output_details):
        print(f"Output {i}:")
        print(f"  Name: {output['name']}")
        print(f"  Shape: {output['shape']}")
        print(f"  Dtype: {output['dtype']}")
        print(f"  Quantization parameters: {output['quantization_parameters']}")
        print(f"  Quantization (scale, zero_point): {output['quantization']}")

if __name__ == '__main__':
    inspect('assets/models/movinet_a2_stream_int8.tflite')
