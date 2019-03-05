//===- CodeEmitVisitor.h --------------------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef TARGET_VANILLA_CODE_EMIT_VISITOR_H
#define TARGET_VANILLA_CODE_EMIT_VISITOR_H
#include <onnc/IR/ComputeVisitor.h>
//#include "NvDlaMeta.h"
namespace onnc {

namespace vanilla {

class CodeEmitVisitor : public ComputeVisitor
{
public:
  //explicit CodeEmitVisitor(NvDlaBackendMeta &meta) noexcept;
  static char ID;

  /// ONNC defined operators @{
  void visit(const Initializer& pInitializer) override;
  void visit(const InputOperator& pInputOperator) override;
  void visit(const OutputOperator& pOutputOperator) override;
  /// @}

  /// ONNX defined operators @{
  void visit(const Conv& pConv) override;
  void visit(const MaxPool& pMaxPool) override;
  void visit(const Relu& pRelu) override;
  void visit(const Softmax& pSoftmax) override;
  void visit(const Reshape& pOp) override;
  void visit(const Concat& pOp) override;
  
  /// @}

  /// ONNC defined operators @{
  void visit(Initializer& pInitializer) override;
  void visit(InputOperator& pInputOperator) override;
  void visit(OutputOperator& pOutputOperator) override;
  /// @}

  /// ONNX defined operators @{
  void visit(Conv& pConv) override;
  void visit (MaxPool& pMaxPool) override;
  void visit(Relu& pRelu) override;
  void visit(Softmax& pSoftmax) override;
  //void visit(Concat& pOp) override;
  //void visit(Reshape& pOp) override;
  /// @}

};

  
} // namespace vanilla
} // namespace onnc

#endif
