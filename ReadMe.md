# ForgeML a TensorFlow C++ Plugin for Unreal Engine
ForgeML is a C++/Python hybrid toolkit designed to simplify the process of training TensorFlow models in Python and running them in C++ with `cppflow`. Allowing for the use of model inference and training in UE at realtime. it handles everything from dataset preparations to model training, validation, checkpointing, and inference.


## Features
- Train in Python
- Automatic Image Loader - Loads images with OpenCV and reshapes them to the correct tensor format.
- Supports Any Model Shape - Easily specify [batch, channels, height, width] or other combinations.
- Inference Wrapper - Run trained TensorFlow models with `cppflow`.
- Model Checkpoints - Save the best performing model automatically.
- ONNX/Keras Loading and Conversions.


## Use Cases
#### Gameplay AI & Procedural Content
- **NPC Behavior Learning**: Train models in Python that learn movement patterns, decision-making, or combat strategies. With real-time inference in UE without Python overhead.
- **Procedural Animation**: Use deep learning to generate animations (e.g. character locomotion, facial expressions) based on context.
- **Level Generation**: Procedurally generate terrain, cities, or puzzles from trained neural networks.

#### Computer Vision in Games & Simulation
- **Object Detection & Tracking**: Train YOLO or custom CNNs in Python, deploy in UE for AR/VR experiences or AI vision systems.
- **Gesture Recognition**: Recognize player gestures from camera input to control gameplay.
- **Visual SLAM Enhancement**: Augment UE's AR tracking with learned features.

#### Simulation & training
- **Robotics Simulation**: Use reinforcement learning in Python, then deploy agents in UE environments for real-time feedback.
- **Driver or Pilot Training**: AI can simulate complex behaviors for traffic, weather, or crowd systems.
- **Industrial Simulations**: Predict failures or optimize processes visually.


## Installation
1) Clone into your Unreal Engine project `Plugins/` folder.
    - Example: `git clone https://github.com/JSzajek/ForgeML.git Plugins/ForgeML`
2) Generate the Visual Studio project files (if necessary).


## Testing
The plugin utilizes Unreal Engine automation framework's SPECs(). It is possible to execute those test in two ways:
1) Command line execution:
   - `UnrealEditor-Cmd.exe <PROJECT_HERE.uproject> -ExecCmds="Automation RunTests ForgeML Unit Test" -unattended -nopause`
2) Editor execution:
   - Open `Tools` -> `Session Frontend`. Navigate to the `Automation` tab.
   - Run the "CLWorks Unit Test" tests.


## **Features**
#### Model Definition and Training
- Build models using TensorFlow/Keras with a consistent, JSON-defined layout structure.
- Automatically supports conversion of `ONNX` to `SavedModel` format.
- Modular `ModelLayout` architecture allows easy customization of layers: Conv1D/2D, MaxPooling, Dense, Dropout, etc.
- Integrated training pipeline for classification tasks with automatic dataset loading and splitting.

#### Model Conversion Utilities
- Converts `ONNX` to TensorFlow `SavedModel`.
  - Conversion chain is ONNX to Keras to SavedModel.
- Extract and exports model meta data including input/output tensor names.
- Label map generation and export to JSON.


#### Image Pre-Processing & Tensor Conversion
- OpenCV based image loader that resized, normalizes, and converts images to any tensor layout.
- Flexible pixel access and image tensor packing based on user-defined shape order.
- SUpport automatic channel conversion.
- Seamless integration with `cppflow::tensor` for model inference.


## Core Classes
### MLModel
Represents a Tensorflow model. Manages the lifetime and loading and training of a Tensorflow model.


### TF Image Loader
Wrapper that takes any `OpenCV` image and automatically reshapes it for TensorFlow models.


## Example Usage
#### Model Creation 
```
TF::MLModel model("<model_name>");

model.AddInput("x", 
			   TF::DataType::Float32,
			   { -1 });

model.AddInput("y", 
			   TF::DataType::Float32,
			   { -1 });

model.AddOutput("add_result");

model.AddLayer(TF::LayerType::Add,
{
	{ "input_names", { "x", "y" } },
	{ "output_name", "add_result" }
});

if (!model.CreateModel())
{
	// Error Creating
}
```

#### Model Inference
```
TF::MLModel::Result results;
if (model.Run(inputs, results))
{
   // Print/Use Results
}
```

#### Image Pre-Processing
```
TF::ImageTensorLoader image_loader(target_width, 
                                   target_height, 
                                   1, 
                                   true, 
                                   TF::ChannelOrder::GrayScale,
                                   TF::ShapeOrder::WidthHeightChannels);


std::unordered_map<std::string, cppflow::tensor> inputs;
if (!image_loader.Load("<insert filepath>", inputs["<input label>"]))
{
	// Error Reading
}
```

### Dependencies
 - Python (3.10.11)
   - Minimum Requirements
     - Tensorflow (2.13)
     - Keras (2.13.1)
     - Onnx2Keras (https://github.com/gmalivenko/onnx2keras)
     - Onnx (1.14.1)
 - C++20 compatible compiler.

### Future Roadmap
- [ ] Embedded Python
- [ ] Model Checkpointing

### License
Licenses Under the **Apache 2.0** License.