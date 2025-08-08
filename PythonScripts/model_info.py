import tensorflow as tf
from tensorflow.python.tools import saved_model_utils
from tensorflow.core.protobuf import saved_model_pb2


def extract_tensor_names(saved_model_dir, signature="serving_default"):

    meta_graph_def = saved_model_utils.get_meta_graph_def(saved_model_dir, tag_set="serve")
    sig_def = meta_graph_def.signature_def[signature]

    io_info = {
        "inputs": {},
        "outputs": {}
    }

    for k, v in sig_def.inputs.items():
        io_info["inputs"][k] = v.name

    for k, v in sig_def.outputs.items():
        io_info["outputs"][k] = v.name

    return io_info




def extract_model_layout(model):

    layout = {
        "model_name": model.name,
        "inputs": [],
        "outputs": [],
        "layers": []
    }

    for input_tensor in model.inputs:
        layout["inputs"].append({
            "name": input_tensor.name.split(":")[0],
            "dtype": input_tensor.dtype.name,
            "shape": [d if d is not None else -1 for d in input_tensor.shape.as_list()],
            "input_type": "data",
        })

    for output_tensor in model.outputs:
        layout["outputs"].append({
            "name": output_tensor.name.split(":")[0]
        })

    for layer in model.layers:
        layer_config = {
            "type": layer.__class__.__name__,
            "params": {
                "input_name": layer.input.name.split(":")[0] if hasattr(layer, "input") else "",
                "output_name": layer.output.name.split(":")[0] if hasattr(layer, "output") else ""
            }
        }
        layout["layers"].append(layer_config)

    return layout