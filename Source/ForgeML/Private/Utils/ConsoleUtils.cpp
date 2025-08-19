#include "Utils/ConsoleUtils.h"

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>

#include "Windows/AllowWindowsPlatformTypes.h"

#include <windows.h>


int ConsoleUtils::Execute(const char* cmd, std::string* output)
{
    SECURITY_ATTRIBUTES sa{ sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE };
    HANDLE hRead, hWrite;
    if (!CreatePipe(&hRead, &hWrite, &sa, 0))
        throw std::runtime_error("Failed to create pipe");

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    si.hStdOutput = hWrite;
    si.hStdError = hWrite;
    si.wShowWindow = SW_HIDE; // hide window

    PROCESS_INFORMATION pi{};

    if (!CreateProcessA(nullptr,
                        (LPSTR)cmd,
                        nullptr,
                        nullptr,
                        TRUE,
                        CREATE_NO_WINDOW,
                        nullptr,
                        nullptr,
                        &si,
                        &pi)) 
    {
        CloseHandle(hRead); CloseHandle(hWrite);
        throw std::runtime_error("Failed to start process");
    }

    CloseHandle(hWrite); // child has it now

    if (output)
    {
        output->clear();

        char buffer[128];
        DWORD bytesRead;
        while (ReadFile(hRead, buffer, sizeof(buffer) - 1, &bytesRead, nullptr) && bytesRead) 
        {
            buffer[bytesRead] = '\0';
            *output += buffer;
        }
    }
    CloseHandle(hRead);

    WaitForSingleObject(pi.hProcess, INFINITE);

    DWORD exitCode;
    GetExitCodeProcess(pi.hProcess, &exitCode);

    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

	return exitCode;
}

#include "Windows/HideWindowsPlatformTypes.h"

#else  // Linux / macOS

#include <cstdio>
#include <memory>
#include <array>
#include <sys/wait.h>

int ConsoleUtils::Execute(const char* cmd, std::string* output)
{
    std::array<char, 256> buffer;
    std::string output;

    // redirect stderr -> stdout with "2>&1"
    std::string fullCmd = cmd + " 2>&1";

    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(fullCmd.c_str(), "r"), pclose);
    if (!pipe)
        throw std::runtime_error("popen failed!");

    if (output)
    {
        output->clear();

        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
            *output += buffer.data();
    }

    int status = pclose(pipe.release());
    int exitCode = WIFEXITED(status) ? WEXITSTATUS(status) : -1;

	return exitCode;
}

#endif