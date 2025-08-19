#pragma once

#include <string>
#include <filesystem>

#include <nlohmann/json.hpp>

namespace TF
{
	/// <summary>
	/// Struct representing the configuration for training a machine learning model.
	/// </summary>
	struct TrainingConfig
	{
	public:
		/// <summary>
		/// Read the training configuration from a JSON file.
		/// </summary>
		/// <param name="filepath">The file path</param>
		void ReadFromFile(const std::filesystem::path& filepath);

		/// <summary>
		/// Write the training configuration to a JSON file.
		/// </summary>
		/// <param name="filepath">The file path</param>
		void WriteToFile(const std::filesystem::path& filepath) const;
	private:
		/// <summary>
		/// Convert the training configuration to a JSON object.
		/// </summary>
		/// <returns>The JSON object</returns>
		nlohmann::json to_json() const;

		/// <summary>
		/// Create a training configuration from a JSON object.
		/// </summary>
		/// <param name="inputJson">The JSON object</param>
		/// <returns>The created training configuration</returns>
		static TrainingConfig from_json(const nlohmann::json& inputJson);
	public:
		// Number of training epochs
		uint32_t epochs = 10;

		// Size of each training batch
		uint32_t batch_size = 32;

		// Learning rate for the optimizer
		float learning_rate = 0.001f;

		// Discount factor for future rewards (used in reinforcement learning)
		float gamma = 0.95f;

		// Whether to shuffle the training data
		bool shuffle = true;

		// Fraction of data to reserve for validation
		float validation_split = 0.0f;


		// TODO:: Implement these options
		//std::string optimizer = "adam";     // Optimizer to use (e.g., "adam", "sgd")
		//std::string loss_function = "mse";  // Loss function to use (e.g., "mse", "categorical_crossentropy")
		//std::vector<std::string> metrics;   // List of metrics to evaluate during training
		//std::string model_save_path;        // Path to save the trained model
		//std::string log_dir;                // Directory for logging training progress
		//bool early_stopping = false;        // Enable early stopping based on validation loss
		//float early_stopping_patience = 5;  // Number of epochs with no improvement before stopping
	};
}