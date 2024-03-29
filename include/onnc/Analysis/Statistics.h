//===- Statistics.h --------------------------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef ONNC_ANALYSIS_STATISTICS_H
#define ONNC_ANALYSIS_STATISTICS_H
#include <onnc/JSON/Storage.h>
#include <onnc/Support/Path.h>
#include <onnc/ADT/StringList.h>
#include <onnc/ADT/Uncopyable.h>
#include <onnc/JSON/Value.h>
#include <ostream>

namespace onnc {

/** \class Statistics
 *  \brief Statistics implements Skymizer's default configuration system.
 *
 *  \code
 *  Statistics config("~/configrc");
 *  StringRef image = config.group("General").readEntry("image", "look.jpg");
 *  \endcode
 */
class Statistics : public json::Storage 
{
public:
  /// Default constructor
  /// Default constructor is invalid because we don't read anything.
  Statistics();

  /// Read the configuration from string @ref pContent
  /// 
  /// @param[in] pContent The content of a configuration file
  explicit Statistics(StringRef pContent);

  /// Read the configuration from string @ref pContent
  /// 
  /// @param[in] pContent The content of a configuration file
  explicit Statistics(const std::string& pContent);

  /// Read the configuration from string @ref pContent
  /// 
  /// @param[in] pContent The content of a configuration file
  explicit Statistics(const char* pContent);

  /// Read the configuration from file @ref pFile
  /// If @ref pFile can not be parsed, then a fatal error is thrown.
  /// 
  /// @param[in] pFile The configuration file
  /// @param[in] pMode If the @ref pFile file is writable, the final result will
  ///                  be writen back when destruction.
  Statistics(const Path& pFile, json::Storage::AccessMode pMode = kReadOnly);

  /// Destructor. If @ref accessMode() is writable, then write back the value.
  virtual ~Statistics() { }

  /// Add a counter.
  /// @retval true Success
  /// @retval false The counter has been added.
  bool addCounter(StringRef pName, StringRef pDesc = "no description");

  /// increase counter, default by 1
  /// @retval true Success
  bool increaseCounter(StringRef pName, unsigned int incNumber=1);

  /// decrease counter, default by 1
  /// @retval true Success
  bool decreaseCounter(StringRef pName, unsigned int decNumber=1);

  /// print counter name, counter value, and its description
  void printCounter(StringRef pName, OStream& pOS);

  /// print all key in Counter group
  /// @retval StringList
  StringList counterList() const;

  /// reset counter as initNum, default is 0
  /// @retval true Success
  bool resetCounter(StringRef pName, int initNum=0);  
};

} // namespace of skymizer

#endif
