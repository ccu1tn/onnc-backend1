//===- CodeEmitVisitor.cpp ------------------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <onnc/Support/IOStream.h>
#include "CodeEmitVisitor.h"
#include <onnc/IR/Compute/Conv.h>
#include <onnc/IR/Compute/MaxPool.h>
#include <onnc/IR/Compute/Relu.h>
#include <onnc/IR/Compute/Softmax.h>
#include <onnc/IR/Compute/Concat.h>
#include <onnc/IR/Compute/Reshape.h>
#include <onnc/IR/Compute/Initializer.h>
#include <onnc/IR/Compute/InputOperator.h>
#include <onnc/IR/Compute/OutputOperator.h>
//#include <stdio.h>

#include  <onnc/IR/Compute/Reshape.h>
#include <onnc/IR/Compute/LRN.h>
#include <onnc/IR/Compute/AveragePool.h>
#include <onnc/IR/Compute/Gemm.h>
#include <onnc/IR/Compute/Softmax.h>
//#include "NvDlaMeta.h"
//#include "fp16.h"

using namespace onnc;
//using namespace onnc::nvdla;
//using namespace onnc;
using namespace std;
using namespace onnc::vanilla;
//using namespace onnc::nvdla;



//using google::protobuf::Message;
//sing google::protobuf::io::CodedInputStream;
//using google::protobuf::io::CodedOutputStream;
//using google::protobuf::io::FileInputStream;
//using google::protobuf::io::FileOutputStream;
//using google::protobuf::io::ZeroCopyInputStream;
//using google::protobuf::io::ZeroCopyOutputStream;

#include <iostream>
#include <string>
#include <cstdio>
#include <stdio.h>
#include <math.h>
#include "arm_nnfunctions.h"
 //#include "arm_math.h"
 double (*foo)(double);      //con trỏ hàm
 void contro()
 {
	foo = &exp;               //trỏ tới hàm exp trong thư viện math.h
	printf("%f",foo(2.0), "\n") ;
 	//return 0;
 }


void create ()
{
  FILE * pFile;
  char buffer[] = {'x','y','z','t',};
  pFile = fopen ("/onnc/onnc-umbrella/Myfile.cpp", "w");
  fwrite (buffer, sizeof(char), sizeof(buffer), pFile);
  fclose (pFile);
  
}

#define NVDLA_LOADABLE_INTERFACE_NONE 0U
#define NVDLA_LOADABLE_INTERFACE_DLA1 1U
enum Interface {
  Interface_NONE = 0,
  Interface_DLA1 = 1,
  //Interface_EMU1 = 2,
  //Interface_MIN = Interface_NONE,
  //Interface_MAX = Interface_EMU1
};
class ILoadable
{
public:

    enum Interface {
        Interface_NONE = NVDLA_LOADABLE_INTERFACE_NONE,
        Interface_DLA1 = NVDLA_LOADABLE_INTERFACE_DLA1,
        //Interface_EMU1 = NVDLA_LOADABLE_INTERFACE_EMU1,
    };
/*
    enum MemoryDomain {
        MemoryDomain_SYSMEM = NVDLA_LOADABLE_MEMORY_DOMAIN_SYSMEM,
        MemoryDomain_SRAM = NVDLA_LOADABLE_MEMORY_DOMAIN_SRAM,
    };

    enum MemoryFlags {
        MemoryFlags_NONE  = NVDLA_LOADABLE_MEMORY_FLAGS_NONE,
        MemoryFlags_ALLOC  = NVDLA_LOADABLE_MEMORY_FLAGS_ALLOC,
        MemoryFlags_SET    = NVDLA_LOADABLE_MEMORY_FLAGS_SET,
        MemoryFlags_INPUT  = NVDLA_LOADABLE_MEMORY_FLAGS_INPUT,
        MemoryFlags_OUTPUT = NVDLA_LOADABLE_MEMORY_FLAGS_OUTPUT,
    };

    enum EventOp {
        EventOp_WAIT   = NVDLA_LOADABLE_EVENT_OP_WAIT,
        EventOp_SIGNAL = NVDLA_LOADABLE_EVENT_OP_SIGNAL
    };

    struct Version
    {
        NvU8 major;
        NvU8 minor;
        NvU8 sub_minor;
        Version(NvU8 maj, NvU8 min, NvU8 sub) : major(maj), minor(min), sub_minor(sub) { }
        Version() : major(0), minor(0), sub_minor(0) { }

        void toC(NvDlaLoadableVersion &c) const
        {
            c.major = major;
            c.minor = minor;
            c.subMinor = sub_minor;
        }
    };*/
};

class NvDlaBackendMeta
{
public:
  NvDlaBackendMeta();

  ~NvDlaBackendMeta();

public:
  // memory allocation information for runtime (firmwares, memory buffer)
  std::vector<ILoadable> m_MemoryListEntries;
  // addresses used in firmware
  //std::vector<ILoadable::AddressListEntry> m_AddressListEntries;
  // input, output specific descriptor
  //std::vector<ILoadable::TensorDescListEntry> m_TensorDescListEntries;
  // relocation information of input/output
  
};
struct MemoryListEntry;
struct MemoryListEntry
    {
        /*NvU16 id;
        NvU64 size;
        NvU32 alignment; // 0 for n/a, otherwise byte alignment
        NvU8  domain;
        static inline NvU8 domain_sysmem() { return MemoryDomain_SYSMEM; }
        static inline NvU8 domain_sram() { return MemoryDomain_SRAM; }
        NvU8  flags; // alloc or alloc_content or is-input or is-output
        static inline NvU8  flags_alloc()  { return MemoryFlags_ALLOC;  }
        static inline NvU8  flags_set()    { return MemoryFlags_SET;    }
        static inline NvU8  flags_input()  { return MemoryFlags_INPUT;  }
        static inline NvU8  flags_output() { return MemoryFlags_OUTPUT; }
        NvU16 bind_id;  // valid iff flag_{input|output}()  is set
        NvU16 tensor_desc_id; // valid iff bind_id is valid ( != -1 )
        std::vector<std::string> contents;  // symbolic reference to content blob
        std::vector<uint64_t>    offsets;   // associated offset for contents

        //MemoryListEntry() : id(0), size(0), alignment(0), domain(0), flags(0),
                            bind_id(0), tensor_desc_id(0), contents(), offsets() { }
        //MemoryListEntry(const MemoryListEntry &o) : id(o.id), size(o.size), alignment(o.alignment), domain(o.domain), flags(o.flags),
                                                    bind_id(o.bind_id),
                                                    tensor_desc_id(o.tensor_desc_id),
                                                    contents(o.contents),
                                                    offsets(o.offsets) { }*/
    };

typedef std::unordered_map<Value *, int> MemoryIdxTable;

void CodeEmitVisitor::visit(const Reshape &pOp)
{
  pOp.print(errs());
  errs() << "\n";

  // Prepare input
  const Tensor *input_data_t = pOp.getInput(0);
  //void *input_data = m_ATable[input_data_t];
  int32_t input_data_ndim = input_data_t->getNumOfDimensions();
  int32_t input_data_dims[4] = {1, 1, 1, 1};
  for (int i = 0; i < input_data_ndim; ++i)
    input_data_dims[i] = input_data_t->dimension(i);
  const Tensor *input_shape_t = pOp.getInput(1);
  //int data_idx = m_pMeta.m_MemIdxTable[(Tensor *)input_data_t];
  //ILoadable::MemoryListEntry data_mle = m_pMeta.m_MemoryListEntries[data_idx];

  //void *input_shape = m_ATable[input_shape_t];
  int32_t input_shape_ndim = input_shape_t->getNumOfDimensions();
  int32_t input_shape_dims[4] = {1, 1, 1, 1};
  for (int i = 0; i < input_shape_ndim; ++i)
    input_shape_dims[i] = input_shape_t->dimension(i);

  // Prepare output
  //void *output_reshaped = m_ATable[output_reshaped_t];
  const Tensor *output_reshaped_t = pOp.getOutput(0);
  int32_t output_reshaped_ndim = output_reshaped_t->getNumOfDimensions();
  int32_t output_reshaped_dims[4] = {1, 1, 1, 1};
  for (int i = 0; i < output_reshaped_ndim; ++i)
    output_reshaped_dims[i] = output_reshaped_t->dimension(i);
  //int out_shape_idx = MemIdxTable[(Tensor *)output_reshaped_t];
  //ILoadable=MemoryListEntries[out_shape_idx];

  //TODO, setup remapping table
  //m_pMeta.m_ReshapeTable[output_reshaped_t] = input_data_t;
}
#define NVDLA_DBG
#define m_pMeta
void CodeEmitVisitor::visit(const Concat &pOp)
 {
  // Prepare input
  int32_t input_inputs_ntensor = pOp.getNumOfInputs() - 0;
  void *input_inputs[input_inputs_ntensor];
  int32_t input_inputs_ndim[input_inputs_ntensor];
  int32_t *input_inputs_dims[input_inputs_ntensor];
  for (int i = 0; i < input_inputs_ntensor; ++i){
    //input_inputs[i] = m_ATable[pOp.getInput(0 + i)];
    input_inputs_ndim[i] = pOp.getInput(0 + i)->getNumOfDimensions();
    input_inputs_dims[i] = new int32_t[4];
    for(int32_t j = 0; j < 4; ++j){
      input_inputs_dims[i][j] = (j < input_inputs_ndim[i]) ? pOp.getInput(0 + i)->dimension(j) : 1;
    }
    NVDLA_DBG("Concat input[%d](%d %d %d %d)\n", i, input_inputs_dims[i][0], input_inputs_dims[i][1], input_inputs_dims[i][2], input_inputs_dims[i][3]);
  }

  // Prepare output
  const Tensor *output_concat_result_t = pOp.getOutput(0);
  //void *output_concat_result = m_ATable[output_concat_result_t];
  int32_t output_concat_result_ndim = output_concat_result_t->getNumOfDimensions();
  int32_t output_concat_result_dims[4] = {1, 1, 1, 1};
  for (int i = 0; i < output_concat_result_ndim; ++i) output_concat_result_dims[i] = output_concat_result_t->dimension(i);
  NVDLA_DBG("Concat output(%d %d %d %d)\n", output_concat_result_dims[0], output_concat_result_dims[1], output_concat_result_dims[2], output_concat_result_dims[3]);
  //int output_mid = m_pMeta.m_MemIdxTable[(Tensor *)output_concat_result_t];
  //ILoadable::= m_MemoryListEntries[output_mid];

  // Prepare attributesinput_inputs_ndim
  int32_t axis = pOp.getAxis().value();
  NVDLA_DBG("Concat AXIS[%d]\n", axis);

  if(axis == 1){

  } else {
    //critical error
  }

  // Clean
  for (int i = 0; i < input_inputs_ntensor; ++i){
    delete [] input_inputs_dims[i];
  }
  errs() << "\n";
  errs() << "\n";
  errs() << "\n";
}


void CodeEmitVisitor::visit(Initializer& pInitializer)
{
//char FILE *file;
//const char *filePath = "/onnc/onnc-umbrella/my_document.txt";
  pInitializer.print(errs());
  errs() << "\n";
  errs() << "you are here \n";
  create();
  errs() << "\n";
  contro();
  //arm_convolve_HWC_q7_basic();
  //writeToFile();
  //ReadProtoFromTextFile(const char *pFilename, pProto);
  //uint8_t fully_connected_run();
}

void CodeEmitVisitor::visit(const Initializer& pInitializer)
{
  pInitializer.print(errs());
  errs() << "you are here \n";
  create();
  errs() << "\n";
  contro();
  //const char *filePath = "/onnc/onnc-umbrella/my_document.txt";
  //writeToFile();
  //uint8_t fully_connected_run();
}

void CodeEmitVisitor::visit(InputOperator& pInputOperator)
{
  pInputOperator.print(errs());
  errs() << "\n";
}

void CodeEmitVisitor::visit(const InputOperator& pInputOperator)
{
  pInputOperator.print(errs());
  errs() << "\n";
}

void CodeEmitVisitor::visit(OutputOperator& pOutputOperator)
{
  pOutputOperator.print(errs());
  errs() << "\n";
}

void CodeEmitVisitor::visit(const OutputOperator& pOutputOperator)
{
  pOutputOperator.print(errs());
  errs() << "\n";
  errs() << "\n";
  errs() << "\n";
}

void CodeEmitVisitor::visit(Conv& pConv)
{
  pConv.print(errs());
  errs() << "\n";
  errs() << "\n";
}

void CodeEmitVisitor::visit(const Conv& pConv)
{
  pConv.print(errs());
  errs() << "\n";
  errs() << "\n";
  errs() << "\n";
  //pOp.print(errs());
  errs() << "\n";

 
}
void CodeEmitVisitor::visit(MaxPool& pMaxPool)
{
  pMaxPool.print(errs());
  errs() << "\n";
}

void CodeEmitVisitor::visit(const MaxPool& pMaxPool)
{
  pMaxPool.print(errs());
  errs() << "\n";
  errs() << "\n";errs() << "\n";
  errs() << "\n";
}
void CodeEmitVisitor::visit(Relu& pRelu)
{
  pRelu.print(errs());
  errs() << "\n";
}

void CodeEmitVisitor::visit(const Relu& pRelu)
{
  pRelu.print(errs());
  errs() << "\n";
  errs() << "\n";
}
void CodeEmitVisitor::visit(Softmax& pSoftmax)
{
  pSoftmax.print(errs());
  errs() << "\n";
}

void CodeEmitVisitor::visit(const Softmax& pSoftmax)
{
  pSoftmax.print(errs());
  errs() << "\n";
  errs() << "\n";
}
