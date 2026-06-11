// GCC 15.2 / MinGW64 time.h compatibility fix
// This file MUST be included before any C++ standard headers

#ifndef GCC15_TIME_FIX_H
#define GCC15_TIME_FIX_H

#ifdef __cplusplus
extern "C" {
#endif

// Include C time.h
#include <time.h>

// Ensure time functions are declared
#ifndef _GLIBCXX_USE_C99
#define _GLIBCXX_USE_C99 1
#endif

#ifdef __cplusplus
}

// Force time types/functions into global namespace for C++
// This works around MinGW64 GCC 15.2 not exporting them properly
#ifndef __GLIBCXX_CTIME_WORKAROUND
#define __GLIBCXX_CTIME_WORKAROUND

// Declare in global namespace if not already there
using ::clock_t;
using ::time_t;
using ::tm;

extern "C" {
    clock_t clock(void);
    double difftime(time_t, time_t);
    time_t mktime(struct tm*);
    time_t time(time_t*);
    char* asctime(const struct tm*);
    char* ctime(const time_t*);
    struct tm* gmtime(const time_t*);
    struct tm* localtime(const time_t*);
    size_t strftime(char*, size_t, const char*, const struct tm*);
}

#endif // __GLIBCXX_CTIME_WORKAROUND

#endif // __cplusplus

#endif // GCC15_TIME_FIX_H
