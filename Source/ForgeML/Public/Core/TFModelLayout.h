#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>

#include <nlohmann/json.hpp>

namespace TF
{
	/// <summary>
	/// Enum representing the data type of an input.
	/// </summary>
	enum class DataType
	{
		Bool,
		UInt8,
		Float32,
		Float64,
		Double,
		Int32,
		Int64,
	};

	/// <summary>
	/// Enum representing the domain type of an input.
	/// </summary>
	enum class DomainType 
	{
		Data,
		Image,
	};

	/// <summary>
	/// Enum representing the type of a layer in a model.
	/// </summary>
	enum class LayerType
	{
		Add,
		Multiply,
		Dense,
		Flatten,
		Activation,
		Dropout,
		Conv1D,
		Conv2D,
		MaxPooling2D,
		BatchNormalization
	};

	/// <summary>
	/// Struct representing the input layer of a model.
	/// </summary>
	struct Input
	{
	public:
		// Desired input node name
		std::string mName;

		DataType mType = DataType::Float32;

		// Shape of the input tensor, use -1 for dynamic dimensions
		std::vector<int> mShape;

		DomainType mDomain = DomainType::Data;
	};

	/// <summary>
	/// Struct representing the output layer of a model.
	/// </summary>
	struct Output
	{
	public:
		// Desired output node name
		std::string mName;
	};

	/// <summary>
	/// Struct representing the layer of a model.
	/// </summary>
	struct Layer
	{
	public:
		// e.g., "Flatten", "Add", "Activation", Dense", "Dropout", "Conv2D", "MaxPooling2D", "BatchNormalization"
		LayerType mType;

		// Generic parameters
		std::unordered_map<std::string, nlohmann::json> mParameters;
	};

	/// <summary>
	/// Struct representing the layout of a ML model.
	/// </summary>
	struct ModelLayout
	{
	public:
		/// <summary>
		/// Read the model layout from a JSON file.
		/// </summary>
		/// <param name="filepath">The file path</param>
		void ReadFromFile(const std::filesystem::path& filepath);

		/// <summary>
		/// Write the model layout to a JSON file.
		/// </summary>
		/// <param name="filepath">The file path</param>
		void WriteToFile(const std::filesystem::path& filepath) const;
	private:
		/// <summary>
		/// Convert the model layout to a JSON object.
		/// </summary>
		/// <returns>The JSON object</returns>
		nlohmann::json to_json() const;

		/// <summary>
		/// Create a model layout from a JSON object.
		/// </summary>
		/// <param name="inputJson">The JSON object</param>
		/// <returns>The created model layout</returns>
		static ModelLayout from_json(const nlohmann::json& inputJson);
	public:
		std::string mModelName;

		std::vector<TF::Input> mInputs;
		std::vector<TF::Output> mOutputs;
		std::vector<TF::Layer> mLayers;
	};
}