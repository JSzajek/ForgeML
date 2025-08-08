#pragma once

#include "Data/IDataBuilder.h"


namespace TF
{
	class FlatFloatDataBuilder : public IDataTensorBuilder<std::vector<float>>
	{
	public:
		FlatFloatDataBuilder(uint32_t size,
							 const std::vector<int64_t>& shape = {});

		virtual void AddInputTensor(const std::string& name, 
									const std::vector<float>& data) override;

		virtual bool CreateTensor(LabeledTensor& output) override;

		virtual std::vector<std::string> GetFeatureNames() override;
	private:
		uint32_t mSize;
		std::vector<int64_t> mShape;

		std::unordered_map<std::string, cppflow::tensor> mInputTensors;
	};
}