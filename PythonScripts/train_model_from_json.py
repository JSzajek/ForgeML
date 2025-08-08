import os
import json
import sys
import tensorflow as tf
import numpy as np

def tf_dtype_from_string(dtype_str):
    return {
        "float32": tf.float32,
        "float64": tf.float64,
        "double": tf.double,
        "int32": tf.int32,
        "int64": tf.int64,
        "uint8": tf.uint8,
        "bool": tf.bool
    }.get(dtype_str, tf.float32)

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def load_image_as_tensor(path, target_shape, dtype):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=target_shape[-1], dtype=dtype, expand_animations=False)
    img = tf.image.resize(img, target_shape[:2])
    img = tf.cast(img, dtype)
    return img

def main(model_path, input_version, output_version, train_config_json, train_data_json):
    # Load files --------------------------------------------------------------
    layout = load_json(f"{model_path}/model_description.json")
    train_config = load_json(train_config_json)
    train_data = load_json(train_data_json)


    # --- Load Keras model ----------------------------------------------------
    input_model_path = f"{model_path}/Saved_{input_version}/"
    model = tf.keras.models.load_model(input_model_path)
    # -------------------------------------------------------------------------


    # --- Prepare inputs and labels -------------------------------------------
    input_data = {}
    for input_spec in layout["inputs"]:
        name = input_spec["name"]
        dtype = tf_dtype_from_string(input_spec["dtype"])
        shape = input_spec["shape"]
        domain = input_spec.get("domain", "data")

        if (domain == "image"):
            print(f"Loading Image Inputs For '{name}'...")

            data_paths = train_data["inputs"][name]
            img_tensors = [
                load_image_as_tensor(path_list[0], shape[1:], dtype)
                for path_list in data_paths
            ]
            tensor = tf.stack(img_tensors)
        else:
            tensor = tf.convert_to_tensor(train_data["inputs"][name], dtype=dtype)

        # Determine expected shape (ignore -1 for batch size)
        target_shape = [dim for dim in shape if dim != -1]
        if target_shape:
            tensor = tf.reshape(tensor, [-1] + target_shape)

        input_data[name] = tensor

    label_data = {}
    for output_spec in layout["outputs"]:
        name = output_spec["name"]
        raw_label = train_data["labels"][name]
        tensor = tf.convert_to_tensor(raw_label, dtype=tf.float32)
        label_data[name] = tensor
    # -------------------------------------------------------------------------


    # --- Compile model -------------------------------------------------------
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    # -------------------------------------------------------------------------

    # --- Fit model -----------------------------------------------------------
    eps = train_config["epochs"]
    b_size = train_config["batch_size"]
    learning_rate = train_config["learning_rate"]
    shuffle = train_config["shuffle"]
    val_split = train_config["validation_split"]

    model.fit(x=input_data, y=label_data, epochs=eps, batch_size=b_size)
    # -------------------------------------------------------------------------

    # --- Save updated model --------------------------------------------------
    output_model_path = f"{model_path}/Saved_{output_version}/"
    model.save(output_model_path)
    print(f"Model Retrained and Saved to {output_model_path}")
    # -------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python train_model.py <model_path> <input_version> <output_version> <train_config.json> <train_data.json>")
        sys.exit(-1)

    model_path = sys.argv[1]
    input_version = sys.argv[2]
    output_version = sys.argv[3]
    train_config_json = sys.argv[4]
    train_data_json = sys.argv[5]

    if (not os.path.exists(f"{model_path}/model_description.json")):
        print("Missing Model Description Json File.")
        sys.exit(-1)

    main(model_path, input_version, output_version, train_config_json, train_data_json)