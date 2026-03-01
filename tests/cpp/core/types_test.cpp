#include <gtest/gtest.h>
#include "edgeunicompile/core/types.h"

using namespace edgeunic;

TEST(DataTypeTest, SizeOfDataType) {
    EXPECT_EQ(GetDataTypeSize(DataType::FLOAT32), 4);
    EXPECT_EQ(GetDataTypeSize(DataType::FLOAT16), 2);
    EXPECT_EQ(GetDataTypeSize(DataType::INT32), 4);
    EXPECT_EQ(GetDataTypeSize(DataType::INT16), 2);
    EXPECT_EQ(GetDataTypeSize(DataType::INT8), 1);
    EXPECT_EQ(GetDataTypeSize(DataType::UINT32), 4);
    EXPECT_EQ(GetDataTypeSize(DataType::UINT16), 2);
    EXPECT_EQ(GetDataTypeSize(DataType::UINT8), 1);
    EXPECT_EQ(GetDataTypeSize(DataType::BOOL), 1);
    EXPECT_EQ(GetDataTypeSize(DataType::COMPLEX64), 8);
}

TEST(DataTypeTest, DataTypeToString) {
    EXPECT_EQ(DataTypeToString(DataType::FLOAT32), "float32");
    EXPECT_EQ(DataTypeToString(DataType::FLOAT16), "float16");
    EXPECT_EQ(DataTypeToString(DataType::INT32), "int32");
    EXPECT_EQ(DataTypeToString(DataType::INT16), "int16");
    EXPECT_EQ(DataTypeToString(DataType::INT8), "int8");
    EXPECT_EQ(DataTypeToString(DataType::UINT32), "uint32");
    EXPECT_EQ(DataTypeToString(DataType::UINT16), "uint16");
    EXPECT_EQ(DataTypeToString(DataType::UINT8), "uint8");
    EXPECT_EQ(DataTypeToString(DataType::BOOL), "bool");
    EXPECT_EQ(DataTypeToString(DataType::COMPLEX64), "complex64");
}

TEST(DataTypeTest, StringToDataType) {
    EXPECT_EQ(StringToDataType("float32"), DataType::FLOAT32);
    EXPECT_EQ(StringToDataType("float16"), DataType::FLOAT16);
    EXPECT_EQ(StringToDataType("int32"), DataType::INT32);
    EXPECT_EQ(StringToDataType("int16"), DataType::INT16);
    EXPECT_EQ(StringToDataType("int8"), DataType::INT8);
    EXPECT_EQ(StringToDataType("uint32"), DataType::UINT32);
    EXPECT_EQ(StringToDataType("uint16"), DataType::UINT16);
    EXPECT_EQ(StringToDataType("uint8"), DataType::UINT8);
    EXPECT_EQ(StringToDataType("bool"), DataType::BOOL);
    EXPECT_EQ(StringToDataType("complex64"), DataType::COMPLEX64);
}

TEST(ShapeTest, NumElements) {
    Shape shape{{2, 3}};
    EXPECT_EQ(shape.NumElements(), 6);

    Shape shape3D{{2, 3, 4}};
    EXPECT_EQ(shape3D.NumElements(), 24);

    Shape scalar{};
    EXPECT_EQ(scalar.NumElements(), 1);
}

TEST(ShapeTest, Rank) {
    Shape scalar{};
    EXPECT_EQ(scalar.Rank(), 0);

    Shape vector{{5}};
    EXPECT_EQ(vector.Rank(), 1);

    Shape matrix{{2, 3}};
    EXPECT_EQ(matrix.Rank(), 2);

    Shape tensor3D{{2, 3, 4}};
    EXPECT_EQ(tensor3D.Rank(), 3);
}

TEST(ShapeTest, ValidShape) {
    EXPECT_TRUE(Shape{{2, 3}}.IsValid());
    EXPECT_TRUE(Shape{{1, 1, 1}}.IsValid());
    EXPECT_FALSE(Shape{{0, 3}}.IsValid());
    EXPECT_FALSE(Shape{{-1, 3}}.IsValid());
    EXPECT_FALSE(Shape{{2, 0}}.IsValid());
}

TEST(ShapeTest, ToString) {
    EXPECT_EQ(Shape{{2, 3}}.ToString(), "[2, 3]");
    EXPECT_EQ(Shape{{1, 1, 1}}.ToString(), "[1, 1, 1]");
    EXPECT_EQ(Shape{{}}.ToString(), "[]");
    EXPECT_EQ(Shape{{10}}.ToString(), "[10]");
}

TEST(OpTypeTest, OpTypeToString) {
    EXPECT_EQ(OpTypeToString(OpType::ADD), "Add");
    EXPECT_EQ(OpTypeToString(OpType::SUBTRACT), "Subtract");
    EXPECT_EQ(OpTypeToString(OpType::MULTIPLY), "Multiply");
    EXPECT_EQ(OpTypeToString(OpType::DIVIDE), "Divide");
    EXPECT_EQ(OpTypeToString(OpType::CONV2D), "Conv2D");
    EXPECT_EQ(OpTypeToString(OpType::MAXPOOL2D), "MaxPool2D");
    EXPECT_EQ(OpTypeToString(OpType::AVERAGEPOOL2D), "AveragePool2D");
    EXPECT_EQ(OpTypeToString(OpType::RELU), "Relu");
}

TEST(OpTypeTest, StringToOpType) {
    EXPECT_EQ(StringToOpType("Add"), OpType::ADD);
    EXPECT_EQ(StringToOpType("add"), OpType::ADD);
    EXPECT_EQ(StringToOpType("Conv2D"), OpType::CONV2D);
    EXPECT_EQ(StringToOpType("conv2d"), OpType::CONV2D);
    EXPECT_EQ(StringToOpType("MaxPool2D"), OpType::MAXPOOL2D);
    EXPECT_EQ(StringToOpType("maxpool2d"), OpType::MAXPOOL2D);
}

TEST(StatusTest, CreateStatus) {
    Status ok = Status::Ok();
    EXPECT_TRUE(ok.IsOk());

    Status error = Status::Error("Test error");
    EXPECT_TRUE(error.IsError());

    EXPECT_EQ(error.GetCode(), StatusCode::ERROR);
    EXPECT_EQ(error.GetMessage(), "Test error");
}

TEST(StatusTest, StatusToString) {
    EXPECT_EQ(Status::Ok().ToString(), "OK");
    EXPECT_EQ(Status::Error("test").ToString(), "ERROR: test");
    EXPECT_EQ(Status::InvalidArgument("invalid").ToString(), "INVALID_ARGUMENT: invalid");
    EXPECT_EQ(Status::NotFound("not found").ToString(), "NOT_FOUND: not found");
}
