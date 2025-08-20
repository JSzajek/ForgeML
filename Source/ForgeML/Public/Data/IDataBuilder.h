#pragma once

#include "Core/TFModelDefines.h"

#include <string>
#include <vector>

namespace TF
{
	/// <summary>
	/// Tensor builder interface for creating input tensors for machine learning models.
	/// </summary>
	/// <typeparam name="T">The data type of tensors</typeparam>
	template<typename T>
	struct IDataTensorBuilder
	{
	public:
		virtual ~IDataTensorBuilder() = default;
	public:
		/// <summary>
		/// Adds an input tensor with the specified name and data.
		/// </summary>
		/// <param name="name">The feature name of the tensor</param>
		/// <param name="data">The data of the tensor</param>
		virtual void AddInputTensor(const std::string& name,
									const T& data) = 0;

		/// <summary>
		/// Creates a labeled tensor from the input data.
		/// </summary>
		/// <param name="output"></param>
		/// <returns></returns>
		virtual bool CreateTensor(LabeledTensor& output) = 0;

		/// <summary>
		/// Retrieves the names of the features in the input data.
		/// </summary>
		/// <returns>The feature names</returns>
		virtual std::vector<std::string> GetFeatureNames() { return {}; }
	};
}