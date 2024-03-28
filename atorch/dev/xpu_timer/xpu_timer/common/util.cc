#include "xpu_timer/common/util.h"

namespace atorch {
namespace util {

std::string execShellCommand(const char* cmd) {
  std::array<char, 128> buffer;
  std::string result;
  FILE* pipe = popen(cmd, "r");
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
    result += buffer.data();
  }
  int returnCode = pclose(pipe);
  if (returnCode != 0) {
    // TODO Handle error
    LOG(FATAL) << "Command failed with return code " << returnCode << std::endl;
  }
  return result;
}

// TODO change into native api when c++20 is ready.
std::vector<std::string> split(const std::string& str,
                               const std::string& delimiter) {
  std::vector<std::string> tokens;
  size_t start = 0;
  size_t end = str.find(delimiter);
  while (end != std::string::npos) {
    tokens.push_back(str.substr(start, end - start));
    start = end + delimiter.length();
    end = str.find(delimiter, start);
  }
  tokens.push_back(str.substr(start, end));
  return tokens;
}

}  // namespace util
}  // namespace atorch
