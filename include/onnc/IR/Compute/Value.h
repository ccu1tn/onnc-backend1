//===- Value.h ------------------------------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef ONNC_IR_COMPUTE_VALUE_H
#define ONNC_IR_COMPUTE_VALUE_H
#include <onnc/IR/Compute/Use.h>
#include <onnc/IR/Compute/Define.h>
#include <onnx/common/ir.h>
#include <vector>
#include <string>

namespace onnc {

/** \class Value
 *  \brief Value represents all values computed by a program.
 *
 *  Value is an object adapter of ::onnx::Tensor.
 */
class Value
{
public:
  typedef std::vector<onnc::Use> UseList;

  enum Type {
    kUndefined = 0,

    // Basic types.
    kFloat,   // float
    kUint8,   // uint8_t
    kInt8,    // int8_t
    kUint16,  // uint16_t
    kInt16,   // int16_t
    kInt32,   // int32_t
    kInt64,   // int64_t
    kString,  // string
    kBoolean, // bool

    // Advanced types
    kFloat16,
    kDouble,
    kUint32,
    kUint64,
    kComplex64,  // complex with float32 real and imaginary components
    kComplex128, // complex with float64 real and imaginary components

    // Basic Tensor types.
    kFloatTensor,   // float
    kUint8Tensor,   // uint8_t
    kInt8Tensor,    // int8_t
    kUint16Tensor,  // uint16_t
    kInt16Tensor,   // int16_t
    kInt32Tensor,   // int32_t
    kInt64Tensor,   // int64_t
    kStringTensor,  // string
    kBooleanTensor, // bool

    // Advanced Tensor types
    kFloat16Tensor,
    kDoubleTensor,
    kUint32Tensor,
    kUint64Tensor,
    kComplex64Tensor,  // complex with float32 real and imaginary components
    kComplex128Tensor  // complex with float64 real and imaginary components
  };

public:
  Value()
    : m_Name(), m_Kind(kUndefined), m_pAdaptee(nullptr) {
  }

  Value(Type pKind)
    : m_Name(), m_Kind(pKind), m_pAdaptee(nullptr) {
  }

  Value(const std::string& pName, Type pKind)
    : m_Name(pName), m_Kind(pKind), m_pAdaptee(nullptr) {
  }

  Value(Type pKind, ::onnx::Tensor& pAdaptee)
    : m_Name(pAdaptee.name()), m_Kind(pKind), m_pAdaptee(&pAdaptee) {
  }

  virtual ~Value() { }

  const std::string& getName() const { return m_Name; }

  Type kind() const { return m_Kind; }

  onnc::Define* getDefine() { return m_pDefine; }

  unsigned int getDefineNo() { return m_DefineNo; }

  const UseList& getUses() const { return m_UseList; }

  UseList& getUses() { return m_UseList; }

  void replaceAllUsesWith(Value *v);

  void adapt(::onnx::Tensor& pAdaptee);

  ::onnx::Tensor* adaptee() { return m_pAdaptee; }

  const ::onnx::Tensor* adaptee() const { return m_pAdaptee; }

  bool hasAdaptee() const { return (nullptr != m_pAdaptee); }

protected:
  onnc::Define* m_pDefine;
  unsigned int m_DefineNo;
  UseList m_UseList;
  std::string m_Name;
  Type m_Kind;

  // Object adaptee. ONNX represents all types in a single tensor.
  ::onnx::Tensor* m_pAdaptee;
};

} // namespace of onnc

#endif