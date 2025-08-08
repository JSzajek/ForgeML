#pragma once

#include "Core/TFModelDefines.h"

#include <string>
#include <vector>

namespace TF
{
	template<typename T>
	struct IDataTensorBuilder
	{
	public:
		virtual ~IDataTensorBuilder() = default;
	public:
		virtual void AddInputTensor(const std::string& name,
									const T& data) = 0;

		virtual bool CreateTensor(LabeledTensor& output) = 0;

		virtual std::vector<std::string> GetFeatureNames() { return {}; }
	};
}