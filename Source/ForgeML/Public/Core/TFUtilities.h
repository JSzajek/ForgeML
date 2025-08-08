#pragma once

#include <string>
#include <sstream>
#include <vector>

#include "CppFlowLib.h"

namespace TF
{
	/// <summary>
	/// Utility function to print the contents of a tensor in a human-readable format.
	/// </summary>
	/// <typeparam name="T">The data type of the tensor</typeparam>
	/// <param name="tensor">The input tensor</param>
	/// <returns>The string form of the tensor</returns>
	template<typename T>
	static std::string PrintTensor(const cppflow::tensor& tensor)
	{
		std::stringstream stream;
		stream << "[";
		std::vector<T> data = tensor.get_data<T>();
		for (uint32_t j = 0; j < data.size(); ++j)
		{
			stream << data[j];
			if (j < data.size() - 1)
				stream << ", ";
		}
		stream << "]";
		return stream.str();
	}
}