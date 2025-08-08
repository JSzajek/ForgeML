#pragma once

#include <unordered_map>
#include <string>

#include "CppFlowLib.h"

namespace TF
{
	using LabeledTensor = std::unordered_map<std::string, cppflow::tensor>;
}