/* Shim for the cmake-generated `x265_4.1/source/x265_config.h.in`.
 *
 * The only field substituted by cmake is X265_BUILD, which `x265.h` uses
 * solely as an API/SONAME version stamp. This crate never calls into the
 * versioned public API, so a fixed value matching `bpg-x265-sys`'s vendored
 * tree (X265_BUILD=215) is enough to let `x265.h` compile.
 */
#ifndef X265_CONFIG_H
#define X265_CONFIG_H

#define X265_BUILD 215

#endif
