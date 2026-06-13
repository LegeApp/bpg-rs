/* 8-bit (`pixel == uint8_t`) instantiation of the stable `extern "C"`
 * boundary over a subset of x265's scalar (`_c`) `EncoderPrimitives`
 * dispatch table -- see `setupCPrimitives` in
 * `x265_4.1/source/common/primitives.cpp`.
 *
 * The function bodies live in `wrapper_impl.h`, shared verbatim with the
 * 10-bit instantiation (`wrapper10.cpp`); this file only sets up the 8-bit
 * `cprims()` accessor and the `bpgprim_` symbol prefix. See `wrapper_impl.h`
 * for the Phase 4 (FULL_RUST_CODEC_PLAN.md) primitive slice it covers and
 * why `setupCPrimitives` itself is not compiled here.
 */
#include <cstring>

#include "common.h"
#include "primitives.h"

namespace X265_NS {
// Forward declarations of the scalar setup routines we link against
// (normally invoked via setupCPrimitives in primitives.cpp).
void setupPixelPrimitives_c(EncoderPrimitives &p);
void setupDCTPrimitives_c(EncoderPrimitives &p);
void setupIntraPrimitives_c(EncoderPrimitives &p);

// `pixel.cpp`'s `extendPicBorder` (unused here, but part of the same
// translation unit) references this global by extern declaration; give it
// a definition so the linker is satisfied. Zero-initialized, never invoked.
EncoderPrimitives primitives;
}

using namespace X265_NS;

namespace {

const EncoderPrimitives &cprims()
{
    static EncoderPrimitives p;
    static bool initialized = false;
    if (!initialized)
    {
        memset(&p, 0, sizeof(p));
        setupPixelPrimitives_c(p);
        setupDCTPrimitives_c(p);
        setupIntraPrimitives_c(p);
        initialized = true;
    }
    return p;
}

} // namespace

#define BPG_PREFIX bpgprim_
#include "wrapper_impl.h"
