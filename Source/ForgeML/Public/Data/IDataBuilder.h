#pragma once

#include <string>
#include <vector>

namespace cppflow
{
	class tensor;
}

namespace TF
{
	template<typename T>
	struct IDataTensorBuilder
	{
	public:
		virtual ~IDataTensorBuilder() = default;
	public:
		virtual cppflow::tensor BuildInputTensor(const UObject* Context) = 0;

		virtual std::vector<std::string> GetFeatureNames() = 0;
	};
}