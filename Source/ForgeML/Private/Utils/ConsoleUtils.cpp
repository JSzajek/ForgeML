#include "Utils/ConsoleUtils.h"

#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <array>

namespace ConsoleUtils
{
	bool Execute(const char* cmd, std::string* output)
	{
		std::array<char, 128> buffer;
		std::string result;
		std::unique_ptr<FILE, decltype(&_pclose)> pipe(_popen(cmd, "r"), _pclose);

		if (!pipe)
		{
			std::cerr << "popen() failed!" << std::endl;
			return false;
		}

		if (output)
		{
			while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) 
				result += buffer.data();
			
			*output = std::move(result);
		}

		return true;
	}
}