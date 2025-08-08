#pragma once

#pragma warning(push)
#pragma warning(disable: 4190)

THIRD_PARTY_INCLUDES_START

#include "../../Common/UndefineMacros_UE.h"

// Disable warning C4190: 'TF_NewWhile' has C-linkage specified, but returns UDT 'TF_WhileParams' which is incompatible with C
#include <cppflow/cppflow.h>

#include "../../Common/RedefineMacros_UE.h"

THIRD_PARTY_INCLUDES_END

#pragma warning(pop)