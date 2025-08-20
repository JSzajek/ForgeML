#pragma once

#include <unordered_map>
#include <string>

#include "CppFlowLib.h"

namespace TF
{
	/// <summary>
	/// Tensors labeled with string keys.
	/// </summary>
	using LabeledTensor = std::unordered_map<std::string, cppflow::tensor>;
}