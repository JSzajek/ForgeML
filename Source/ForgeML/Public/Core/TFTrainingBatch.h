#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>

#include <nlohmann/json.hpp>

namespace TF
{
	/// <summary>
	/// Struct representing a training input.
	/// </summary>
	struct NamedInput
	{
		// Must match ModelLayout.input.name
		std::string mName;

		// Outer vector = batch, inner = flat feature array
		std::vector<nlohmann::json> mData;
	};

	/// <summary>
	/// Struct representing a training label.
	/// </summary>
	struct NamedLabel
	{
		// Must match ModelLayout.output.name
		std::string mName;

		// e.g., one-hot vectors
		std::vector<nlohmann::json> mData;
	};

	/// <summary>
	/// Struct representing a training label.
	/// </summary>
	struct RewardData
	{
		// The current observed state (could be a tensor, vector, etc.)
		nlohmann::json mState;
		
		// The action taken (one-hot, discrete ID, or continuous vector)
		nlohmann::json mAction;       
		
		// The reward value from this step
		float mReward = 0.0f;
		
		// Optional: The next state after taking the action
		nlohmann::json mNextState;
	};

	/// <summary>
	/// Struct representing a training batch that is labeled for supervised training.
	/// </summary>
	struct LabeledTrainingBatch
	{
	public:
		inline operator bool() const
		{
			return !mInputs.empty() && !mLabels.empty();
		}
	public:
		inline void Clear()
		{
			mInputs.clear();
			mLabels.clear();
		}

		/// <summary>
		/// Read the training batch from a JSON file.
		/// </summary>
		/// <param name="filepath">The file path</param>
		void ReadFromFile(const std::filesystem::path& filepath);

		/// <summary>
		/// Write the training batch to a JSON file.
		/// </summary>
		/// <param name="filepath">The file path</param>
		void WriteToFile(const std::filesystem::path& filepath) const;
	private:
		/// <summary>
		/// Convert the training batch to a JSON object.
		/// </summary>
		/// <returns>The JSON object</returns>
		nlohmann::json to_json() const;

		/// <summary>
		/// Create a training batch from a JSON object.
		/// </summary>
		/// <param name="inputJson">The JSON object</param>
		/// <returns>The created training batch</returns>
		static LabeledTrainingBatch from_json(const nlohmann::json& inputJson);
	public:
		std::vector<NamedInput> mInputs;
		std::vector<NamedLabel> mLabels;
	};

	/// <summary>
	/// Struct representing a training batch that has data corresponding to a 
	/// reward value for supervised training.
	/// </summary>
	struct RewardTrainingBatch
	{
	public:
		inline operator bool() const
		{
			return !mSamples.empty();
		}
	public:
		inline void Clear()
		{
			mSamples.clear();
		}

		/// <summary>
		/// Read the training batch from a JSON file.
		/// </summary>
		/// <param name="filepath">The file path</param>
		void ReadFromFile(const std::filesystem::path& filepath);

		/// <summary>
		/// Write the training batch to a JSON file.
		/// </summary>
		/// <param name="filepath">The file path</param>
		void WriteToFile(const std::filesystem::path& filepath) const;
	private:
		/// <summary>
		/// Convert the training batch to a JSON object.
		/// </summary>
		/// <returns>The JSON object</returns>
		nlohmann::json to_json() const;

		/// <summary>
		/// Create a training batch from a JSON object.
		/// </summary>
		/// <param name="inputJson">The JSON object</param>
		/// <returns>The created training batch</returns>
		static RewardTrainingBatch from_json(const nlohmann::json& inputJson);
	public:
		std::vector<RewardData> mSamples;
	};
}