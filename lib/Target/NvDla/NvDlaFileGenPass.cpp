//===- InterpreterPass.cpp ------------------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "NvDlaFileGenPass.h"

#include <onnc/IR/Compute/Tensor.h>
#include <onnc/IR/Compute/Initializer.h>
#include <onnc/IR/Compute/InputOperator.h>
#include <onnc/IR/Compute/OutputOperator.h>
#include <onnc/Support/Casting.h>
#include <onnc/Support/IOStream.h>
#include <onnc/Support/Timer.h>

using namespace onnc;

//===----------------------------------------------------------------------===//
// NvDlaFileGenPass
//===----------------------------------------------------------------------===//
NvDlaFileGenPass::NvDlaFileGenPass(TargetBackend *pBackend,
                  NvDlaBackendMeta *pMeta)
  : ModulePass(ID),
    m_pBackend(pBackend), m_pMeta(pMeta){
}

Pass::ReturnType NvDlaFileGenPass::runOnModule(Module &pModule)
{
  //file output
  //priv::LoadableFactory::LoadablePrivPair loadable = priv::LoadableFactory::newLoadable();
  m_pMeta->m_Loadable.priv()->setMemoryListEntries(m_pMeta->m_MemoryListEntries);
  m_pMeta->m_Loadable.priv()->setTensorDescListEntries(m_pMeta->m_TensorDescListEntries);
  m_pMeta->m_Loadable.priv()->setAddressListEntries(m_pMeta->m_AddressListEntries);
  m_pMeta->m_Loadable.priv()->setEventListEntries(m_pMeta->m_EventListEntries);
  m_pMeta->m_Loadable.priv()->setTaskListEntries(m_pMeta->m_TaskListEntries);
  m_pMeta->m_Loadable.priv()->setSubmitListEntries(m_pMeta->m_SubmitListEntries);
  m_pMeta->m_Loadable.priv()->serialize();
  return Pass::kModuleNoChanged;
}


//===----------------------------------------------------------------------===//
// Factory method
//===----------------------------------------------------------------------===//
char NvDlaFileGenPass::ID = 0;

NvDlaFileGenPass *onnc::CreateNvDlaFileGenPass(TargetBackend *pBackend,
                                              NvDlaBackendMeta *pMeta) {
  return new NvDlaFileGenPass(pBackend, pMeta);
}
