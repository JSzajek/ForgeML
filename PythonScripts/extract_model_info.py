import tensorflow as tf
import json
import sys
import os
from model_info import extract_tensor_names
from model_info import extract_model_layout

def extract_model_info(model_path):
	# Extract Signatures
    io = extract_tensor_names(model_path)
    with open(model_path + "/cppflow_io_names.json", "w") as f:
        json.dump(io, f, indent=4)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_model_info.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]

    extract_model_info(model_path)