#include "Core/TFTrainingBatch.h"

#include <fstream>

namespace TF
{
	void TrainingBatch::ReadFromFile(const std::filesystem::path& filepath)
	{
		std::ifstream ifs(filepath);
		if (!ifs)
			throw std::runtime_error("Failed to open file for reading: " + filepath.string());

		nlohmann::json j;
		ifs >> j;
		TrainingBatch batch = from_json(j);

		mInputs = std::move(batch.mInputs);
		mLabels = std::move(batch.mLabels);
	}

	void TrainingBatch::WriteToFile(const std::filesystem::path& filepath) const
	{
		std::filesystem::path parent_path = filepath.parent_path();
		if (!std::filesystem::is_directory(parent_path) || !std::filesystem::exists(parent_path))
			std::filesystem::create_directory(parent_path);

		std::ofstream ofs(filepath);
		if (!ofs)
			throw std::runtime_error("Failed to open file for writing: " + filepath.string());

		ofs << to_json().dump(4);
	}

	nlohmann::json TrainingBatch::to_json() const
	{
		nlohmann::json result;
		for (const auto& input : mInputs)
			result["inputs"][input.mName].push_back(input.mData);

		for (const auto& label : mLabels)
			result["labels"][label.mName].push_back(label.mData);

		return result;
	}

	TrainingBatch TrainingBatch::from_json(const nlohmann::json& inputJson)
	{
		TrainingBatch batch;

		for (auto& [key, val] : inputJson["inputs"].items())
		{
			batch.mInputs.push_back(
			{ 
				key, 
				val.get<std::vector<nlohmann::json>>() 
			});
		}

		for (auto& [key, val] : inputJson["labels"].items())
		{
			batch.mLabels.push_back(
			{ 
				key, 
				val.get<std::vector<nlohmann::json>>() 
			});
		}
		return batch;
	}
}