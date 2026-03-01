#include <gtest/gtest.h>
#include "edgeunicompile/core/context.h"

using namespace edgeunic;

TEST(ContextTest, CreateContext) {
    auto ctx = Context::Create();
    ASSERT_TRUE(ctx);
    EXPECT_TRUE(ctx->GetOptLevel() == 3);
    EXPECT_TRUE(ctx->GetSramSize() == 32 * 1024 * 1024);  // 32MB
    EXPECT_TRUE(ctx->GetTargetArch() == "armv8");
    EXPECT_TRUE(ctx->GetDebugMode() == false);
    EXPECT_TRUE(ctx->GetVerboseMode() == false);
}

TEST(ContextTest, SetAndGetAttributes) {
    auto ctx = Context::Create();

    ctx->SetAttribute("test_key", 42);
    EXPECT_TRUE(ctx->GetAttribute<int>("test_key") == 42);

    ctx->SetAttribute("pi", 3.14159f);
    EXPECT_TRUE(ctx->GetAttribute<float>("pi") == 3.14159f);

    ctx->SetAttribute("name", "test");
    EXPECT_TRUE(ctx->GetAttribute<std::string>("name") == "test");

    EXPECT_TRUE(ctx->GetAttribute<std::string>("nonexistent", "default") == "default");
}

TEST(ContextTest, PerformanceCounters) {
    auto ctx = Context::Create();

    ctx->IncrementCounter("counter1");
    EXPECT_TRUE(ctx->GetCounter("counter1") == 1);

    ctx->IncrementCounter("counter1", 5);
    EXPECT_TRUE(ctx->GetCounter("counter1") == 6);

    ctx->IncrementCounter("counter2", 10);
    EXPECT_TRUE(ctx->GetCounter("counter2") == 10);

    auto all = ctx->GetAllCounters();
    EXPECT_TRUE(all.size() == 2);
}

TEST(ContextTest, Configuration) {
    auto ctx = Context::Create();

    ctx->SetOptLevel(0);
    EXPECT_TRUE(ctx->GetOptLevel() == 0);

    ctx->SetOptLevel(4);
    EXPECT_TRUE(ctx->GetOptLevel() == 4);

    ctx->SetSramSize(16 * 1024 * 1024);  // 16MB
    EXPECT_TRUE(ctx->GetSramSize() == 16 * 1024 * 1024);

    ctx->SetWorkloadSize(1024);  // 1KB
    EXPECT_TRUE(ctx->GetWorkloadSize() == 1024);

    ctx->SetTargetArch("riscv");
    EXPECT_TRUE(ctx->GetTargetArch() == "riscv");

    ctx->SetDebugMode(true);
    EXPECT_TRUE(ctx->GetDebugMode() == true);

    ctx->SetVerboseMode(true);
    EXPECT_TRUE(ctx->GetVerboseMode() == true);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
