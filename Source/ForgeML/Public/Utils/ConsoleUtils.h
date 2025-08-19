#pragma once

#include <string>

struct FORGEML_API ConsoleUtils
{
public:
	/// <summary>
	/// Executes a console command and optionally captures the output.
	/// </summary>
	/// <param name="cmd">The console command to execute</param>
	/// <param name="output">The captured output or nullptr if no output is to be captured</param>
	/// <returns>The exit code</returns>
	static int Execute(const char* cmd,
					   std::string* output = nullptr);
};