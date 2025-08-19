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

def discounted_returns(rewards, gamma):
    """Compute discounted returns for a list of rewards (per episode)."""
    returns = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns

def train_supervised(model, layout, train_config, train_data):
    print("Supervised Training Detected...")

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
    eps = train_config.get("epochs", 1)
    b_size = train_config.get("batch_size", 32)
    learning_rate = train_config.get("learning_rate", 1e-3)
    shuffle = train_config["shuffle"]
    val_split = train_config["validation_split"]

    model.fit(x=input_data, y=label_data, epochs=eps, batch_size=b_size)
    # -------------------------------------------------------------------------


def train_with_reward(model, layout, train_config, train_data):
    print("Reward-Based Training Detected...")

    # --- Check model output dimension ---
    test_input_shape = model.input_shape
    output_shape = model.output_shape

    # Multiple outputs (not allowed)
    if isinstance(output_shape, list):
        raise ValueError("Model has multiple outputs. Vanilla Q-learning requires a single scalar output.")
    if output_shape[-1] != 1:
        raise ValueError(f"Model output dimension must be 1 (scalar Q-value), but got {output_shape[-1]}.")


    lr = train_config.get("learning_rate", 1e-3)
    gamma = train_config.get("gamma", 0.95)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="mse")

    states = []
    actions = []
    rewards = []
    next_states = []


    # Parse JSON into numpy arrays
    for sample in train_data:
        states.append(sample["state"])
        actions.append(sample["action"])
        rewards.append(sample["reward"])

        if "next_state" in sample:
            next_states.append(sample["next_state"])


    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    rewards = np.array(rewards, dtype=np.float32)
    
    # Optional next_state, next_action, done
    next_states = np.array([d.get("next_state") if "next_state" in d else None for d in train_data], dtype=object)
    next_actions = np.array([d.get("next_action", a) for d, a in zip(train_data, actions)], dtype=np.float32)
    dones = np.array([d.get("done", 0.0) for d in train_data], dtype=np.float32)

    # Compute next Q-values safely
    q_next = np.zeros_like(rewards)
    has_next = [ns is not None for ns in next_states]
    if any(has_next):
        # Only compute for samples that have next_state
        valid_indices = [i for i, h in enumerate(has_next) if h]
        valid_next_states = np.array([next_states[i] for i in valid_indices], dtype=np.float32)
        valid_next_actions = np.array([next_actions[i] for i in valid_indices], dtype=np.float32)
        q_next_vals = model.predict([valid_next_states, valid_next_actions], verbose=0).squeeze()
        q_next[valid_indices] = q_next_vals

    # Q-learning targets
    targets = rewards + gamma * (1 - dones) * q_next
    targets = targets.reshape(-1, 1)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,        # how many epochs with no improvement before stopping
        restore_best_weights=True
    )

    # Now just fit the model with (states, targets)
    history = model.fit(
        states,
        targets,
        epochs=train_config.get("epochs", 1),
        batch_size=train_config.get("batch_size", 32),
        verbose=1,
        shuffle=True,
        callbacks=[early_stop]
    )

    print(f"[RL Training] Final Loss: {history.history['loss'][-1]:.4f} from {len(states)} samples")



def main(model_path, input_version, output_version):
    # Load files --------------------------------------------------------------
    layout = load_json(f"{model_path}/model_description.json")
    train_config = load_json(f"{model_path}/train/train_config.json")

    has_supervised_data = os.path.exists(f"{model_path}/train/s-train_data.json")
    has_reward_data = os.path.exists(f"{model_path}/train/r-train_data.json")

    s_train_data = load_json(f"{model_path}/train/s-train_data.json") if has_supervised_data else None
    r_train_data = load_json(f"{model_path}/train/r-train_data.json") if has_reward_data else None
    
    # --- Load Keras model ----------------------------------------------------
    input_model_path = f"{model_path}/Saved_{input_version}/"
    model = tf.keras.models.load_model(input_model_path)
    # -------------------------------------------------------------------------

    if (has_supervised_data):
        train_supervised(model, layout, train_config, s_train_data)
    
    if (has_reward_data):
        train_with_reward(model, layout, train_config, r_train_data)
    

    # --- Save updated model --------------------------------------------------
    output_model_path = f"{model_path}/Saved_{output_version}/"
    model.save(output_model_path)
    print(f"Model Retrained and Saved to {output_model_path}")
    # -------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python train_model.py <model_path> <input_version> <output_version>")
        sys.exit(-1)

    model_path = sys.argv[1]
    input_version = sys.argv[2]
    output_version = sys.argv[3]

    if (not os.path.exists(f"{model_path}/model_description.json")):
        print("Missing Model Description Json File.")
        sys.exit(-1)

    if (not os.path.exists(f"{model_path}/train/train_config.json")):
        print("Missing Training Config Json File.")
        sys.exit(-1)

    if (not os.path.exists(f"{model_path}/train/s-train_data.json") and
        not os.path.exists(f"{model_path}/train/r-train_data.json")):
        print("Missing Training Data Json File.")
        sys.exit(-1)

    main(model_path, input_version, output_version)