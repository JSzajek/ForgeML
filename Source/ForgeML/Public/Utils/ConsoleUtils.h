#pragma once

#include <string>

namespace ConsoleUtils
{
	/// <summary>
	/// Executes a console command and optionally captures the output.
	/// </summary>
	/// <param name="cmd">The console command to execute</param>
	/// <param name="output">The captured output or nullptr if no output is to be captured</param>
	/// <returns>True if the execution is successful</returns>
	static bool Execute(const char* cmd, 
						std::string* output = nullptr);
}