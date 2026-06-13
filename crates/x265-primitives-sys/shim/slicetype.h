/* Shim for `x265_4.1/source/encoder/slicetype.h`.
 *
 * `common/pixel.cpp` includes "slicetype.h" only for the `LOWRES_COST_MASK`
 * macro (used by a lookahead weighted-prediction helper that this crate
 * never calls). The real `slicetype.h` drags in `slice.h`/`motion.h`/
 * `piclist.h`/`threadpool.h`/`temporalfilter.h` -- the whole video-encoder
 * dependency graph, which is out of scope for the still-image primitive
 * boundary. This crate's include path puts `shim/` ahead of
 * `x265_4.1/source/encoder/`, so the quoted `#include "slicetype.h"`
 * resolves here instead.
 */
#ifndef X265_SLICETYPE_H
#define X265_SLICETYPE_H

#define LOWRES_COST_MASK ((1 << 14) - 1)

#endif
