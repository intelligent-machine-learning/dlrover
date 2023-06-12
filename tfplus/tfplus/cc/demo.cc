// Copyright 2023 tfplus.

#include <ctime>
#include <iostream>

#include "tfplus/cc/demo.h"

namespace tfplus {

void Demo::print_localtime() {
    std::time_t result = std::time(nullptr);
    std::cout << std::asctime(std::localtime(&result));
}

}
