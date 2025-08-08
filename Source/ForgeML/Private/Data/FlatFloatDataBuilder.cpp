#include "Data/FlatFloatDataBuilder.h"

namespace TF
{
	FlatFloatDataBuilder::FlatFloatDataBuilder(uint32_t size,
											   const std::vector<int64_t>& shape)
		: mSize(size),
		mShape(shape)
	{
	}

	void FlatFloatDataBuilder::AddInputTensor(const std::string& name, 
											  const std::vector<float>& data)
	{
		std::vector<float> localData(data);

		if (localData.size() != mSize)
			localData.resize(mSize, 0);

		if (mShape.empty())
		{
			mInputTensors[name] = localData;
		}
		else if (mShape.size() == 1 && mShape[0] == -1)
		{
			mInputTensors[name] = cppflow::tensor(localData, { static_cast<int64_t>(localData.size()) });
		}
		else if (mShape.size() == 1 && mShape[0] != static_cast<int64_t>(localData.size()))
		{
			std::cerr << "Shape mismatch for tensor '" << name << "'. Expected size: " << mShape[0] << ", got: " << localData.size() << std::endl;
			return;
		}
		else
		{
			mInputTensors[name] = cppflow::tensor(localData, mShape);
		}
	}

	bool FlatFloatDataBuilder::CreateTensor(LabeledTensor& output)
	{
		if (mInputTensors.empty())
		{
			std::cerr << "No input tensors have been added." << std::endl;
			return false;
		}

		for (const auto& [key, tensor] : mInputTensors)
			output[key] = tensor;

		return true;
	}

	std::vector<std::string> FlatFloatDataBuilder::GetFeatureNames()
	{
		std::vector<std::string> featureNames;
		featureNames.reserve(mInputTensors.size());

		for (const auto& [key, tensor] : mInputTensors)
			featureNames.push_back(key);
		
		return featureNames;
	}

}