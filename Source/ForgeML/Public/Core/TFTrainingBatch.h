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
	/// Struct representing a training batch.
	/// </summary>
	struct TrainingBatch
	{
	public:
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
		static TrainingBatch from_json(const nlohmann::json& inputJson);
	public:
		std::vector<NamedInput> mInputs;
		std::vector<NamedLabel> mLabels;
	};
}