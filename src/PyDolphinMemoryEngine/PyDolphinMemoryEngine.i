 %module PyDolphinMemoryEngine
 %{
 /* Includes the header in the wrapper code */
 #include "Dolphin-memory-engine/Source/MemoryWatch/MemoryWatch.h"
 #include "Dolphin-memory-engine/Source/MemoryScanner/MemoryScanner.h"
 #include "Dolphin-memory-engine/Source/DolphinProcess/DolphinAccessor.h"
 #include "Dolphin-memory-engine/Source/DolphinProcess/IDolphinProcess.h"
 #include "Dolphin-memory-engine/Source/DolphinProcess/Linux/LinuxDolphinProcess.h"
 #include "Dolphin-memory-engine/Source/Common/CommonTypes.h"
 #include "Dolphin-memory-engine/Source/Common/CommonUtils.h"
 #include "Dolphin-memory-engine/Source/Common/MemoryCommon.h"
 %}
 
 /* Parse the header file to generate wrappers */
 %include "Dolphin-memory-engine/Source/MemoryWatch/MemoryWatch.h"
 %include "Dolphin-memory-engine/Source/MemoryScanner/MemoryScanner.h"
 %include "Dolphin-memory-engine/Source/DolphinProcess/DolphinAccessor.h"
 %include "Dolphin-memory-engine/Source/DolphinProcess/IDolphinProcess.h"
 %include "Dolphin-memory-engine/Source/DolphinProcess/Linux/LinuxDolphinProcess.h"
 %include "Dolphin-memory-engine/Source/Common/CommonTypes.h"
 %include "Dolphin-memory-engine/Source/Common/CommonUtils.h"
 %include "Dolphin-memory-engine/Source/Common/MemoryCommon.h"
