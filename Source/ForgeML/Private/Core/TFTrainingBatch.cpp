#include "Core/TFTrainingBatch.h"

#include <fstream>

namespace TF
{
	void LabeledTrainingBatch::ReadFromFile(const std::filesystem::path& filepath)
	{
		std::ifstream ifs(filepath);
		if (!ifs)
			throw std::runtime_error("Failed to open file for reading: " + filepath.string());

		nlohmann::json j;
		ifs >> j;
		LabeledTrainingBatch batch = from_json(j);

		mInputs = std::move(batch.mInputs);
		mLabels = std::move(batch.mLabels);
	}

	void LabeledTrainingBatch::WriteToFile(const std::filesystem::path& filepath) const
	{
		std::filesystem::path parent_path = filepath.parent_path();
		if (!std::filesystem::is_directory(parent_path) || !std::filesystem::exists(parent_path))
			std::filesystem::create_directory(parent_path);

		std::ofstream ofs(filepath);
		if (!ofs)
			throw std::runtime_error("Failed to open file for writing: " + filepath.string());

		ofs << to_json().dump(4);
	}

	nlohmann::json LabeledTrainingBatch::to_json() const
	{
		nlohmann::json result;
		for (const auto& input : mInputs)
			result["inputs"][input.mName].push_back(input.mData);

		for (const auto& label : mLabels)
			result["labels"][label.mName].push_back(label.mData);

		return result;
	}

	LabeledTrainingBatch LabeledTrainingBatch::from_json(const nlohmann::json& inputJson)
	{
		LabeledTrainingBatch batch;

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

	void RewardTrainingBatch::ReadFromFile(const std::filesystem::path& filepath)
	{
		std::ifstream ifs(filepath);
		if (!ifs)
			throw std::runtime_error("Failed to open file for reading: " + filepath.string());

		nlohmann::json j;
		ifs >> j;
		RewardTrainingBatch batch = from_json(j);

		mSamples = std::move(batch.mSamples);
	}

	void RewardTrainingBatch::WriteToFile(const std::filesystem::path& filepath) const
	{
		std::filesystem::path parent_path = filepath.parent_path();
		if (!std::filesystem::is_directory(parent_path) || !std::filesystem::exists(parent_path))
			std::filesystem::create_directory(parent_path);

		std::ofstream ofs(filepath);
		if (!ofs)
			throw std::runtime_error("Failed to open file for writing: " + filepath.string());

		ofs << to_json().dump(4);
	}

	nlohmann::json RewardTrainingBatch::to_json() const
	{
		nlohmann::json result;

		for (const auto& sample : mSamples)
		{
			nlohmann::json entry;

			entry["state"] = sample.mState;
			entry["action"] = sample.mAction;
			entry["reward"] = sample.mReward;

			result.emplace_back(entry);
		}

		return result;
	}

	RewardTrainingBatch RewardTrainingBatch::from_json(const nlohmann::json& inputJson)
	{
		RewardTrainingBatch batch;

		for (const auto& entry : inputJson)
		{
			RewardData data;

			// state is an array, so push each JSON object into mSamples
			for (const auto& stateJson : entry.at("state"))
			{
				data.mState.push_back(stateJson);
			}

			data.mAction = entry.at("action");
			data.mReward = entry.at("reward").get<float>();

			batch.mSamples.push_back(std::move(data));
		}

		return batch;
	}
}