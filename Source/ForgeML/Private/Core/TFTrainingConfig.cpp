#include "Core/TFTrainingConfig.h"

#include <fstream>

namespace TF
{
	void TrainingConfig::ReadFromFile(const std::filesystem::path& filepath)
	{
		std::ifstream ifs(filepath);
		if (!ifs)
			throw std::runtime_error("Failed to open File for Reading: " + filepath.string());

		nlohmann::json j;
		ifs >> j;
		TrainingConfig batch = from_json(j);
	}

	void TrainingConfig::WriteToFile(const std::filesystem::path& filepath) const
	{
		std::filesystem::path parent_path = filepath.parent_path();
		if (!std::filesystem::is_directory(parent_path) || !std::filesystem::exists(parent_path))
			std::filesystem::create_directory(parent_path);

		std::ofstream ofs(filepath);
		if (!ofs)
			throw std::runtime_error("Failed to open File for Writing: " + filepath.string());

		ofs << to_json().dump(4);
	}

	nlohmann::json TrainingConfig::to_json() const
	{
		nlohmann::json result;

		result["epochs"] = epochs;
		result["batch_size"] = batch_size;
		result["learning_rate"] = learning_rate;
		result["shuffle"] = shuffle;
		result["validation_split"] = validation_split;
		// Add other fields as needed

		return result;
	}

	TrainingConfig TrainingConfig::from_json(const nlohmann::json& inputJson)
	{
		TrainingConfig config;

		if (inputJson.contains("epochs"))
			config.epochs = inputJson["epochs"].get<uint32_t>();
		if (inputJson.contains("batch_size"))
			config.batch_size = inputJson["batch_size"].get<uint32_t>();
		if (inputJson.contains("learning_rate"))
			config.learning_rate = inputJson["learning_rate"].get<float>();
		if (inputJson.contains("shuffle"))
			config.shuffle = inputJson["shuffle"].get<bool>();
		if (inputJson.contains("validation_split"))
			config.validation_split = inputJson["validation_split"].get<float>();
		// Add other fields as needed

		return config;
	}
}