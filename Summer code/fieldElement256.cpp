#include "fieldElement256.hpp"

#ifdef USE_FIELD_256

namespace virgo256 {

// Static member definitions
bool fieldElement256::initialized = false;
int fieldElement256::multCounter = 0;
int fieldElement256::addCounter = 0;
bool fieldElement256::isCounting = false;
bool fieldElement256::isSumchecking = false;

} // namespace virgo256

#endif
