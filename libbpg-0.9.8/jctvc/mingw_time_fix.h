/* Workaround for GCC 15.x/MinGW64 time.h compatibility issue */
#ifndef MINGW_TIME_FIX_H
#define MINGW_TIME_FIX_H

#if defined(__MINGW64__) || defined(__MINGW32__)
/* Include C headers before any C++ headers to ensure proper declarations */
#include <time.h>
#include <wchar.h>
#include <sys/types.h>

/* Force time functions into global namespace for C++ <ctime> compatibility */
#ifdef __cplusplus
namespace std {
    using ::clock_t;
    using ::time_t;
    using ::tm;
    using ::clock;
    using ::difftime;
    using ::mktime;
    using ::time;
    using ::asctime;
    using ::ctime;
    using ::gmtime;
    using ::localtime;
    using ::strftime;
}
#endif

#endif /* __MINGW64__ || __MINGW32__ */

#endif /* MINGW_TIME_FIX_H */
