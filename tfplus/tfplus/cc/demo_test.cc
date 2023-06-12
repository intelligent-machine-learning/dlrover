// Copyright 2023 tfplus.

#include "tfplus/cc/demo.h"
#include "gtest/gtest.h"


namespace {

TEST(DemoTest, DemoBasicTest) {
    tfplus::Demo dm;
    dm.print_localtime();
    EXPECT_EQ(true, true);
}

}  // namespace
