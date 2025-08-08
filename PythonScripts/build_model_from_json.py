# build_model_from_json.py
import tensorflow as tf
import json
import sys
from model_info import extract_tensor_names
from model_info import extract_model_layout
from pathlib import Path
from tensorflow.python.tools import saved_model_utils
from tensorflow.core.protobuf import saved_model_pb2


def build_model(layout):
    inputs = {}
    for inp in layout["inputs"]:
        inputs[inp["name"]] = tf.keras.Input(shape=inp["shape"][1:], name=inp["name"], dtype=inp["dtype"])

    layers = dict(inputs)
    for layer in layout["layers"]:
        typ = layer["type"]
        params = layer["params"]
        if typ == "Add":
            layers[params["output_name"]] = tf.keras.layers.Add(
                name=params["output_name"]
            )([layers[n] for n in params["input_names"]])
        if typ == "Multiply":
            layers[params["output_name"]] = tf.keras.layers.Multiply(
                name=params["output_name"]
            )([layers[n] for n in params["input_names"]])
        elif typ == "Dense":
            layers[params["output_name"]] = tf.keras.layers.Dense(
                units=params["units"],
                activation=params.get("activation", None),
                name=params["output_name"]
            )(layers[params["input_name"]])
        elif typ == "Flatten":
            layers[params["output_name"]] = tf.keras.layers.Flatten(
                name=params["output_name"]
            )(layers[params["input_name"]])
        elif typ == "Activation":
            layers[params["output_name"]] = tf.keras.layers.Activation(
                activation=params["activation"],
                name=params["output_name"]
            )(layers[params["input_name"]])
        elif typ == "Dropout":
            layers[params["output_name"]] = tf.keras.layers.Dropout(
                rate=params["rate"],
                name=params["output_name"]
            )(layers[params["input_name"]])
        elif typ == "Conv1D":
            layers[params["output_name"]] = tf.keras.layers.Conv1D(
                filters=params["filters"],
                kernel_size=params["kernel_size"],
                strides=params.get("strides", 1),
                padding=params.get("padding", "valid"),
                activation=params.get("activation", None),
                name=params["output_name"]
            )(layers[params["input_name"]])
        elif typ == "Conv2D":
            layers[params["output_name"]] = tf.keras.layers.Conv2D(
                filters=params["filters"],
                kernel_size=tuple(params["kernel_size"]),
                strides=tuple(params.get("strides", [1, 1])),
                padding=params.get("padding", "valid"),
                activation=params.get("activation", None),
                name=params["output_name"]
            )(layers[params["input_name"]])
        elif typ == "MaxPooling2D":
            layers[params["output_name"]] = tf.keras.layers.MaxPooling2D(
                pool_size=tuple(params.get("pool_size", [2, 2])),
                strides=tuple(params.get("strides", [2, 2])),
                padding=params.get("padding", "valid"),
                name=params["output_name"]
            )(layers[params["input_name"]])
        elif typ == "BatchNormalization":
            layers[params["output_name"]] = tf.keras.layers.BatchNormalization(
                name=params["output_name"]
            )(layers[params["input_name"]])
        # TODO:: Extend with more layer types...

    outputs = [layers[o["name"]] for o in layout["outputs"]]
    model = tf.keras.Model(inputs=list(inputs.values()), outputs=outputs)
    return model


def main(output_path, version):
    print(f"Creating Model Description At [{output_path}]...")
    json_path = f"{output_path}/model_description.json"
    with open(json_path, "r") as f:        
        layout = json.load(f)

    #model_name = layout["model_name"]

    model = build_model(layout)

    print(f"Creating Model...")
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    saved_model_path = f"{output_path}/Saved_{version}/"

    model.save(saved_model_path)
    print(f"Saved Model to: {saved_model_path}")

    # Extract Signatures
    io = extract_tensor_names(saved_model_path)

    ionames_path = f"{saved_model_path}/cppflow_io_names.json"
    with open(ionames_path, "w") as f:
        json.dump(io, f, indent=2)

if __name__ == "__main__":
    arg_cnt = len(sys.argv)
    if arg_cnt != 3:
        print("Usage: python build_model_from_json.py <model_path> <version>")
        sys.exit(-1)

    output_path = sys.argv[1]
    version = sys.argv[2]

    main(output_path, version)
