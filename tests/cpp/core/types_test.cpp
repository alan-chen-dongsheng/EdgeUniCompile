#include <gtest/gtest.h>
#include "edgeunicompile/core/types.h"

using namespace edgeunic;

TEST(DataTypeTest, SizeOfDataType) {
    EXPECT_EQ(GetDataTypeSize(DataType::kFloat32), 4u);
    EXPECT_EQ(GetDataTypeSize(DataType::kFloat16), 2u);
    EXPECT_EQ(GetDataTypeSize(DataType::kInt32), 4u);
    EXPECT_EQ(GetDataTypeSize(DataType::kInt16), 2u);
    EXPECT_EQ(GetDataTypeSize(DataType::kInt8), 1u);
    EXPECT_EQ(GetDataTypeSize(DataType::kUInt32), 4u);
    EXPECT_EQ(GetDataTypeSize(DataType::kUInt16), 2u);
    EXPECT_EQ(GetDataTypeSize(DataType::kUInt8), 1u);
    EXPECT_EQ(GetDataTypeSize(DataType::kBool), 1u);
    EXPECT_EQ(GetDataTypeSize(DataType::kComplex64), 8u);
}

TEST(DataTypeTest, DataTypeToString) {
    EXPECT_EQ(DataTypeToString(DataType::kFloat32), "float32");
    EXPECT_EQ(DataTypeToString(DataType::kFloat16), "float16");
    EXPECT_EQ(DataTypeToString(DataType::kInt32), "int32");
    EXPECT_EQ(DataTypeToString(DataType::kInt16), "int16");
    EXPECT_EQ(DataTypeToString(DataType::kInt8), "int8");
    EXPECT_EQ(DataTypeToString(DataType::kUInt32), "uint32");
    EXPECT_EQ(DataTypeToString(DataType::kUInt16), "uint16");
    EXPECT_EQ(DataTypeToString(DataType::kUInt8), "uint8");
    EXPECT_EQ(DataTypeToString(DataType::kBool), "bool");
    EXPECT_EQ(DataTypeToString(DataType::kComplex64), "complex64");
}

TEST(DataTypeTest, StringToDataType) {
    EXPECT_EQ(StringToDataType("float32"), DataType::kFloat32);
    EXPECT_EQ(StringToDataType("float16"), DataType::kFloat16);
    EXPECT_EQ(StringToDataType("int32"), DataType::kInt32);
    EXPECT_EQ(StringToDataType("int16"), DataType::kInt16);
    EXPECT_EQ(StringToDataType("int8"), DataType::kInt8);
    EXPECT_EQ(StringToDataType("uint32"), DataType::kUInt32);
    EXPECT_EQ(StringToDataType("uint16"), DataType::kUInt16);
    EXPECT_EQ(StringToDataType("uint8"), DataType::kUInt8);
    EXPECT_EQ(StringToDataType("bool"), DataType::kBool);
    EXPECT_EQ(StringToDataType("complex64"), DataType::kComplex64);
}

TEST(ShapeTest, NumElements) {
    Shape shape{{2, 3}};
    EXPECT_EQ(shape.NumElements(), 6u);

    Shape shape3D{{2, 3, 4}};
    EXPECT_EQ(shape3D.NumElements(), 24u);

    Shape scalar{};
    EXPECT_EQ(scalar.NumElements(), 1u);
}

TEST(ShapeTest, Rank) {
    Shape scalar{};
    EXPECT_EQ(scalar.Rank(), 0u);

    Shape vector{{5}};
    EXPECT_EQ(vector.Rank(), 1u);

    Shape matrix{{2, 3}};
    EXPECT_EQ(matrix.Rank(), 2u);

    Shape tensor3D{{2, 3, 4}};
    EXPECT_EQ(tensor3D.Rank(), 3u);
}

TEST(ShapeTest, ValidShape) {
    EXPECT_TRUE((Shape{{2, 3}}.IsValid()));
    EXPECT_TRUE((Shape{{1, 1, 1}}.IsValid()));
    EXPECT_FALSE((Shape{{0, 3}}.IsValid()));
    EXPECT_FALSE((Shape{{-1, 3}}.IsValid()));
    EXPECT_FALSE((Shape{{2, 0}}.IsValid()));
}

TEST(ShapeTest, ToString) {
    EXPECT_EQ((Shape{{2, 3}}.ToString()), "[2, 3]");
    EXPECT_EQ((Shape{{1, 1, 1}}.ToString()), "[1, 1, 1]");
    EXPECT_EQ((Shape{{}}.ToString()), "[]");
    EXPECT_EQ((Shape{{10}}.ToString()), "[10]");
}

TEST(OpTypeTest, OpTypeToString) {
    EXPECT_EQ(OpTypeToString(OpType::kAdd), "Add");
    EXPECT_EQ(OpTypeToString(OpType::kSubtract), "Subtract");
    EXPECT_EQ(OpTypeToString(OpType::kMultiply), "Multiply");
    EXPECT_EQ(OpTypeToString(OpType::kDivide), "Divide");
    EXPECT_EQ(OpTypeToString(OpType::kConv2D), "Conv2D");
    EXPECT_EQ(OpTypeToString(OpType::kMaxPool2D), "MaxPool2D");
    EXPECT_EQ(OpTypeToString(OpType::kAveragePool2D), "AveragePool2D");
    EXPECT_EQ(OpTypeToString(OpType::kRelu), "Relu");
}

TEST(OpTypeTest, StringToOpType) {
    EXPECT_EQ(StringToOpType("Add"), OpType::kAdd);
    EXPECT_EQ(StringToOpType("add"), OpType::kAdd);
    EXPECT_EQ(StringToOpType("Conv2D"), OpType::kConv2D);
    EXPECT_EQ(StringToOpType("conv2d"), OpType::kConv2D);
    EXPECT_EQ(StringToOpType("MaxPool2D"), OpType::kMaxPool2D);
    EXPECT_EQ(StringToOpType("maxpool2d"), OpType::kMaxPool2D);
}

TEST(StatusTest, CreateStatus) {
    Status ok = Status::Ok();
    EXPECT_TRUE(ok.IsOk());

    Status error = Status::Error("Test error");
    EXPECT_TRUE(error.IsError());

    EXPECT_EQ(error.Code(), StatusCode::kError);
    EXPECT_EQ(error.Message(), "Test error");
}

TEST(StatusTest, StatusToString) {
    EXPECT_EQ(Status::Ok().ToString(), "OK");
    EXPECT_EQ(Status::Error("test").ToString(), "ERROR: test");
    EXPECT_EQ(Status::InvalidArgument("invalid").ToString(), "INVALID_ARGUMENT: invalid");
    EXPECT_EQ(Status::NotFound("not found").ToString(), "NOT_FOUND: not found");
}
