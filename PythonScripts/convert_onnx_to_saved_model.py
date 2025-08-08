import onnx
import os
import argparse
from onnx2keras import onnx_to_keras
import tensorflow as tf


def extract_input_info(onnx_model):
    initializer_names = {init.name for init in onnx_model.graph.initializer}
    inputs = []
    for inp in onnx_model.graph.input:
        if inp.name not in initializer_names:
            shape = [dim.dim_value if dim.dim_value > 0 else None
                     for dim in inp.type.tensor_type.shape.dim]
            inputs.append((inp.name, shape))
    return inputs
 

def convert_onnx_to_keras(onnx_path):
    onnx_model = onnx.load(onnx_path)
    input_info = extract_input_info(onnx_model)

    input_names = [name for name, _ in input_info]
    input_shapes = [tuple(shape[1:]) for _, shape in input_info]  # Exclude batch dim

    print("Input Names: ")
    print(input_names)
    
    print("Input Shapes: ")
    print(input_shapes)

    keras_model = onnx_to_keras(
        onnx_model,
        input_names=input_names,
        #input_shapes=input_shapes,
        name_policy='renumerate'
    )
    return keras_model
 
def save_as_saved_model(keras_model, output_dir):

    tf.saved_model.save(keras_model, output_dir)
    print(f"SavedModel saved to: {output_dir}")
 
 
 
def main():
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorFlow SavedModel")
    parser.add_argument("onnx_model", help="Path to the ONNX model file")
    parser.add_argument("output_dir", help="Directory to save the SavedModel")

    args = parser.parse_args()

    keras_model = convert_onnx_to_keras(args.onnx_model)
    print("Created Keras Model")
     
    save_as_saved_model(keras_model, args.output_dir)


if __name__ == "__main__":
    main()
 
