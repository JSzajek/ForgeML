#pragma once

#include "Data/IDataBuilder.h"

namespace TF
{
	/// <summary>
	/// FlatFloatDataBuilder is a concrete implementation 
	/// of IDataTensorBuilder for float vectors.
	/// </summary>
	class FORGEML_API FlatFloatDataBuilder : public IDataTensorBuilder<std::vector<float>>
	{
	public:
		/// <summary>
		/// Constructor initializing a FlatFloatDataBuilder.
		/// </summary>
		/// <param name="size">The size of the data</param>
		/// <param name="shape">The shape of the tensor</param>
		FlatFloatDataBuilder(uint32_t size,
							 const std::vector<int64_t>& shape = {});
	public:
		/// <summary>
		/// Adds an input tensor with the specified name and data.
		/// </summary>
		/// <param name="name">The feature name of the tensor</param>
		/// <param name="data">The data of the tensor</param>
		virtual void AddInputTensor(const std::string& name, 
									const std::vector<float>& data) override;

		/// <summary>
		/// Creates a labeled tensor from the input data.
		/// </summary>
		/// <param name="output"></param>
		/// <returns></returns>
		virtual bool CreateTensor(LabeledTensor& output) override;

		/// <summary>
		/// Retrieves the names of the features in the input data.
		/// </summary>
		/// <returns>The feature names</returns>
		virtual std::vector<std::string> GetFeatureNames() override;
	private:
		uint32_t mSize;
		std::vector<int64_t> mShape;

		std::unordered_map<std::string, cppflow::tensor> mInputTensors;
	};
}