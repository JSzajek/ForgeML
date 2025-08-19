#pragma once

#include "Core/TFModelDefines.h"
#include "Core/TFUtilities.h"
#include "Core/TFModelLayout.h"
#include "Core/TFTrainingBatch.h"
#include "Core/TFTrainingConfig.h"

#include <vector>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <mutex>

namespace TF
{
	/// <summary>
	/// Class representing a Machine Learning Model that can be used for training and inference.
	/// </summary>
	class FORGEML_API MLModel
	{
	public:
		/// <summary>
		/// Constructor initializing a MLModel.
		/// </summary>
		/// <param name="modelname">The model name</param>
		/// <param name="output">The output path</param>
		MLModel(const std::string& modelname,
				const std::filesystem::path& output = "");

		/// <summary>
		/// Default destructor.
		/// </summary>
		~MLModel();
	public:
		/// <summary>
		/// Loads Pre-Trained Models. Currently Only Supports loading ONNX or SavedModel format models.
		/// 
		/// Non-SavedModel formats will be converted to SavedModel format.
		/// 
		/// The model will be created/saved in-place unless an output is given.
		/// </summary>
		/// <param name="loadpath">The filepath to load from</param>
		/// <param name="output">The output directory override</param>
		/// <returns>True if the load was successful</returns>
		bool LoadFrom(const std::filesystem::path& loadpath,
					  const std::filesystem::path& output = "");

		/// <summary>
		/// Checks whether the current model exists.
		/// </summary>
		/// <returns>True if the model exists</returns>
		bool DoesModelExists();

		/// <summary>
		/// Loads the model at a specific version number if it exists.
		/// Or the latest if -1 is passed.
		/// </summary>
		/// <param name="version">The version number</param>
		/// <returns>True if the model exists and loads</returns>
		bool LoadIfExists(int32_t version = -1);

		/// <summary>
		/// Adds an input to the model.
		/// </summary>
		/// <param name="name">The name of the input</param>
		/// <param name="dtype">The data type of the input</param>
		/// <param name="shape">The shape of the input</param>
		/// <param name="domain">The domain type of the input</param>
		void AddInput(const std::string& name,
					  DataType dtype,
					  std::vector<int> shape, 
					  DomainType domain = DomainType::Data);

		/// <summary>
		/// Adds an output to the model.
		/// </summary>
		/// <param name="name">The output label</param>
		void AddOutput(const std::string& name);

		/// <summary>
		/// Adds a layer to the model.
		/// </summary>
		/// <param name="type">The type of the layer</param>
		/// <param name="params">The parameters of the layer</param>
		void AddLayer(LayerType type, 
					  const std::unordered_map<std::string, nlohmann::json>& params);

		/// <summary>
		/// Adds supervised training data to the model.
		/// </summary>
		/// <param name="input_name">The input name of the training batch</param>
		/// <param name="input_values">The input values</param>
		/// <param name="label_name">The label name of the training batch</param>
		/// <param name="label_outputs">The label outputs</param>
		void AddSupervisedTrainingData(const std::string& input_name, 
									   const nlohmann::json& input_values,
									   const std::string& label_name,
									   const nlohmann::json& label_outputs);

		void AddRewardData(const nlohmann::json& state_values,
						   const nlohmann::json& action_values,
						   float reward,
						   const nlohmann::json& next_state_values = {});

		/// <summary>
		/// Save the model layout to a JSON file.
		/// </summary>
		/// <param name="path">The output path of the json file</param>
		void SaveLayoutJson(const std::filesystem::path& path) const;

		/// <summary>
		/// Save the training data to a JSON file.
		/// </summary>
		/// <param name="path">The output path of the json file</param>
		void SaveTrainingJson(const std::filesystem::path& path) const;

		/// <summary>
		/// Creates the model based on the current layout and training data.
		/// </summary>
		/// <returns>True if the creation was successful</returns>
		bool CreateModel();

		/// <summary>
		/// Launches the training of the model.
		/// </summary>
		/// <param name="output_path">The output path of the model</param>
		/// <param name="epochs">The number of epochs</param>
		/// <param name="batchSize">The batch size</param>
		/// <param name="learning_rate">The learning rate</param>
		/// <param name="shuffle">Whether to shuffle the training</param>
		/// <param name="validation_split">The validation split</param>
		/// <param name="clean_data">Whether to clear the training data after this training session</param>
		/// <returns>True if the training was successful</returns>
		bool TrainModel(uint32_t epochs = 10,
						uint32_t batchSize = 32,
						float learning_rate = 0.001f,
						float gamma = 0.95f,
						bool shuffle = true,
						float validation_split = 0.0f,
						bool clean_data = true);

		/// <summary>
		/// Runs the model with the given input tensors and returns the output.
		/// </summary>
		/// <param name="input_tensors">The input tensors</param>
		/// <param name="output">The output result</param>
		/// <returns>True if the running the model was successful</returns>
		bool Run(const LabeledTensor& input_tensors,
				 LabeledTensor& output);

		/// <summary>
		/// Exports all of the model's components to the specified directory.
		/// </summary>
		/// <param name="directory">The output directory</param>
		void ExportAll(const std::filesystem::path& directory) const;
	private:
		/// <summary>
		/// Converts the model to a SavedModel format if it is not already in that format.
		/// </summary>
		/// <param name="filepath">The input model's filepath</param>
		/// <param name="outputpath">The output model path</param>
		/// <returns>True if conversion was successful</returns>
		bool ConvertModelToSavedModel(const std::filesystem::path& filepath,
									  const std::filesystem::path& outputpath);

		/// <summary>
		/// Retrieves the root directory for the model based on the output directory and model name.
		/// </summary>
		/// <returns>The model root directory</returns>
		std::string GetModelRoot() const;

		/// <summary>
		/// Creates a model name based on the model number based on the input parameter.
		/// If the version is -1, it will use the current model version.
		/// </summary>
		/// <param name="version">Version number override</param>
		/// <returns>The version model name</returns>
		std::string CreateModelName(int32_t version = -1) const;
	public:
		std::string mName;
		std::atomic<uint32_t> mModelVersion = 0;

		std::unique_ptr<cppflow::model> mpModel = nullptr;
		std::mutex mModelMutex = {};

		std::string mScriptDirectory;
		std::string mOutputDirectory;


		ModelLayout mLayout;

		std::unordered_map<std::string, std::string> mInputToIONamesMap;
		std::unordered_map<std::string, std::string> mOutputIONamesMap;
		std::vector<std::string> mOutputIONames;

		LabeledTrainingBatch mSupervisedTrainingBatch;
		RewardTrainingBatch mRewardTrainingBatch;
	};
}