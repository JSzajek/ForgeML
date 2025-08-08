#include "Core/TFModelLayout.h"

#include <iostream>
#include <fstream>

namespace TF
{
	std::string DataTypeToString(DataType type)
	{
		switch (type)
		{
		case DataType::Bool:
			return "bool";
		case DataType::UInt8:
			return "uint8";
		case DataType::Float32:
			return "float32";
		case DataType::Float64:
			return "float64";
		case DataType::Double:
			return "double";
		case DataType::Int32:
			return "int32";
		case DataType::Int64:
			return "int64";
		default:
			throw std::invalid_argument("Unsupported DataType");
		}
		return "";
	}

	DataType StringToDataType(const std::string& str)
	{
		if (str == "bool")
			return DataType::Bool;
		else if (str == "uint8")
			return DataType::UInt8;
		else if (str == "float32")
			return DataType::Float32;
		else if (str == "float64")
			return DataType::Float64;
		else if (str == "double")
			return DataType::Double;
		else if (str == "int32")
			return DataType::Int32;
		else if (str == "int64")
			return DataType::Int64;
		else
			throw std::invalid_argument("Unsupported DataType String: " + str);

		return DataType::Float32; // Default Fallback
	}

	std::string DomainTypeToString(DomainType type)
	{
		switch (type)
		{
		case DomainType::Data:
			return "data";
		case DomainType::Image:
			return "image";
		default:
			throw std::invalid_argument("Unsupported DomainType");
		}
		return "";
	}

	DomainType StringToDomainType(const std::string& str)
	{
		if (str == "data")
			return DomainType::Data;
		else if (str == "image")
			return DomainType::Image;
		else
			throw std::invalid_argument("Unsupported DomainType String: " + str);

		return DomainType::Data; // Default Fallback
	}

	std::string LayerTypeToString(LayerType type)
	{
		switch (type)
		{
			case LayerType::Add:
				return "Add";
			case LayerType::Multiply:
				return "Multiply";
			case LayerType::Dense:
				return "Dense";
			case LayerType::Flatten:
				return "Flatten";
			case LayerType::Activation:
				return "Activation";
			case LayerType::Dropout:
				return "Dropout";
			case LayerType::Conv1D:
				return "Conv1D";
			case LayerType::Conv2D:
				return "Conv2D";
			case LayerType::MaxPooling2D:
				return "MaxPooling2D";
			case LayerType::BatchNormalization:
				return "BatchNormalization";
			default:
				throw std::invalid_argument("Unsupported LayerType");
		}
		return "";
	}

	LayerType StringToLayerType(const std::string& str)
	{
		if (str == "Add")
			return LayerType::Add;
		else if (str == "Multiply")
			return LayerType::Multiply;
		else if (str == "Dense")
			return LayerType::Dense;
		else if (str == "Flatten")
			return LayerType::Flatten;
		else if (str == "Activation")
			return LayerType::Activation;
		else if (str == "Dropout")
			return LayerType::Dropout;
		else if (str == "Conv1D")
			return LayerType::Conv1D;
		else if (str == "Conv2D")
			return LayerType::Conv2D;
		else if (str == "MaxPooling2D")
			return LayerType::MaxPooling2D;
		else if (str == "BatchNormalization")
			return LayerType::BatchNormalization;
		else
			throw std::invalid_argument("Unsupported LayerType String: " + str);

		return LayerType::Dense; // Default Fallback
	}


	void ModelLayout::ReadFromFile(const std::filesystem::path& filepath)
	{
		std::ifstream ifs(filepath);
		if (!ifs)
			throw std::runtime_error("Failed to open file for reading: " + filepath.string());

		nlohmann::json j;
		ifs >> j;
		ModelLayout layout = from_json(j);

		mModelName	= layout.mModelName;
		mInputs		= std::move(layout.mInputs);
		mOutputs	= std::move(layout.mOutputs);
		mLayers		= std::move(layout.mLayers);
	}

	void ModelLayout::WriteToFile(const std::filesystem::path& filepath) const
	{
		std::filesystem::path parent_path = filepath.parent_path();
		if (!std::filesystem::is_directory(parent_path) || !std::filesystem::exists(parent_path))
			std::filesystem::create_directory(parent_path);

		nlohmann::json j = to_json();

		std::ofstream ofs(filepath);
		if (!ofs)
			throw std::runtime_error("Failed to open file for writing: " + filepath.string());

		ofs << to_json().dump(4);
	}


	nlohmann::json ModelLayout::to_json() const
	{
		nlohmann::json result;
		result["model_name"] = mModelName;

		for (const auto& input : mInputs)
		{
			const std::string type_str = DataTypeToString(input.mType);
			const std::string domain_str = DomainTypeToString(input.mDomain);

			result["inputs"].push_back(
			{
				{ "name", input.mName },
				{ "dtype", type_str },
				{ "shape", input.mShape },
				{ "domain", domain_str }
			});
		}

		for (const auto& output : mOutputs)
		{
			result["outputs"].push_back(
			{
				{ "name", output.mName }
			});
		}

		for (const auto& layer : mLayers)
		{
			const std::string layer_type = LayerTypeToString(layer.mType);

			result["layers"].push_back(
			{
				{ "type", layer_type },
				{ "params", layer.mParameters }
			});
		}

		return result;
	}

	ModelLayout ModelLayout::from_json(const nlohmann::json& inputJson)
	{
		ModelLayout layout;
		layout.mModelName = inputJson.at("model_name");

		for (const auto& jin : inputJson.at("inputs"))
		{
			layout.mInputs.push_back(
			{ 
				jin.at("name"), 
				StringToDataType(jin.at("dtype")),
				jin.at("shape").get<std::vector<int>>(),
				StringToDomainType(jin.value("input_type", "data")) // Default to "data" if not specified
			});

		}

		for (const auto& jout : inputJson.at("outputs"))
		{
			layout.mOutputs.push_back(
			{ 
				jout.at("name") 
			});
		}

		for (const auto& jlayer : inputJson.at("layers"))
		{
			layout.mLayers.push_back(
			{ 
				jlayer.at("type"), 
				jlayer.at("params") 
			});
		}

		return layout;
	}
}