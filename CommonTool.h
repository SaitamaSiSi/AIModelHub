#pragma once

#ifdef _WIN32
#include <direct.h>
#elif __APPLE__ || __linux__
#include<unistd.h>
#endif
#include <string>

class CommonTool
{
public:
	std::string GetCurrentPath() {
		char runPath[1024] = { 0 };
		std::ignore = _getcwd(runPath, sizeof(runPath));
		return runPath;
	}
};

