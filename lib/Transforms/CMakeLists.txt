add_circt_library(CIRCTTransforms
  ConvertVectorTypeToHWArrayType.cpp
  FlattenMemRefs.cpp
  StripDebugInfoWithPred.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Transforms

  LINK_LIBS PUBLIC
  CIRCTControlFlowLoopAnalysis
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms

  DEPENDS
  CIRCTTransformsPassIncGen
)
