#include <gtest/gtest.h>
#include "edgeunicompile/ir/tensor.h"

using namespace edgeunic;

TEST(TensorTest, CreateTensor) {
    Tensor tensor("test_tensor", DataType::FLOAT32, Shape{{2, 3}});
    EXPECT_EQ(tensor.GetName(), "test_tensor");
    EXPECT_EQ(tensor.GetDataType(), DataType::FLOAT32);
    EXPECT_EQ(tensor.GetShape().dims, std::vector<int64_t>({2, 3}));
    EXPECT_TRUE(tensor.GetData().empty());
    EXPECT_FALSE(tensor.IsConstant());
    EXPECT_TRUE(tensor.GetProducerNode().empty());
}

TEST(TensorTest, SetAndGetName) {
    Tensor tensor("test_tensor", DataType::FLOAT32, Shape{{2, 3}});
    EXPECT_EQ(tensor.GetName(), "test_tensor");

    tensor.SetName("new_name");
    EXPECT_EQ(tensor.GetName(), "new_name");
}

TEST(TensorTest, SetAndGetDataType) {
    Tensor tensor("test_tensor", DataType::FLOAT32, Shape{{2, 3}});
    EXPECT_EQ(tensor.GetDataType(), DataType::FLOAT32);

    tensor.SetDataType(DataType::INT32);
    EXPECT_EQ(tensor.GetDataType(), DataType::INT32);
}

TEST(TensorTest, SetAndGetShape) {
    Tensor tensor("test_tensor", DataType::FLOAT32, Shape{{2, 3}});
    EXPECT_EQ(tensor.GetShape().dims, std::vector<int64_t>({2, 3}));

    tensor.SetShape(Shape{{3, 4, 5}});
    EXPECT_EQ(tensor.GetShape().dims, std::vector<int64_t>({3, 4, 5}));
}

TEST(TensorTest, NumElements) {
    Tensor tensor1("tensor1", DataType::FLOAT32, Shape{{2, 3}});
    EXPECT_EQ(tensor1.NumElements(), 6);

    Tensor tensor2("tensor2", DataType::FLOAT32, Shape{{2, 3, 4}});
    EXPECT_EQ(tensor2.NumElements(), 24);
}

TEST(TensorTest, IsConstant) {
    Tensor tensor("test_tensor", DataType::FLOAT32, Shape{{2, 3}});
    EXPECT_FALSE(tensor.IsConstant());

    tensor.SetData(std::vector<uint8_t>(8));
    EXPECT_TRUE(tensor.IsConstant());
}

TEST(TensorTest, DataOperations) {
    Tensor tensor("test_tensor", DataType::FLOAT32, Shape{{2, 2}});  // 2x2 tensor = 4 elements = 16 bytes (32-bit)

    std::vector<uint8_t> data(16, 0);
    tensor.SetData(data);
    EXPECT_EQ(tensor.GetData().size(), 16);

    auto& tensorData = tensor.GetData();
    EXPECT_EQ(tensorData.size(), 16);

    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(data[i], tensorData[i]);
    }
}

TEST(TensorTest, ProducerNode) {
    Tensor tensor("test_tensor", DataType::FLOAT32, Shape{{2, 3}});
    EXPECT_TRUE(tensor.GetProducerNode().empty());

    tensor.SetProducerNode("node1");
    EXPECT_EQ(tensor.GetProducerNode(), "node1");
}

TEST(TensorTest, IsValid) {
    Tensor valid_tensor("test_tensor", DataType::FLOAT32, Shape{{2, 3}});
    EXPECT_TRUE(valid_tensor.IsValid());

    Tensor invalid_dtype("invalid_dtype", DataType::UNKNOWN, Shape{{2, 3}});
    EXPECT_FALSE(invalid_dtype.IsValid());

    Tensor invalid_shape("invalid_shape", DataType::FLOAT32, Shape{{-1, 3}});
    EXPECT_FALSE(invalid_shape.IsValid());
}

TEST(TensorTest, ToString) {
    Tensor tensor("test_tensor", DataType::FLOAT32, Shape{{2, 3}});
    std::string str = tensor.ToString();

    EXPECT_TRUE(str.find("test_tensor") != std::string::npos);
    EXPECT_TRUE(str.find("float32") != std::string::npos);
    EXPECT_TRUE(str.find("[2, 3]") != std::string::npos);
}
