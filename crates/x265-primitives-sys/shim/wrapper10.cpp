/* 10-bit (`pixel == uint16_t`) instantiation of the stable `extern "C"`
 * boundary over x265's scalar (`_c`) `EncoderPrimitives` dispatch table.
 *
 * Compiled under `HIGH_BIT_DEPTH=1`, `X265_DEPTH=10`, `X265_NS=x265_10bit`
 * (see build.rs), so `pixel == uint16_t` and x265's bit-depth-specialized
 * scalar primitives are linked. The function bodies are shared verbatim
 * with the 8-bit instantiation (`wrapper.cpp`) via `wrapper_impl.h`; this
 * file only sets up the 10-bit `cprims()` accessor and the `bpgprim10_`
 * symbol prefix, so the full 8-bit primitive surface is exposed at 10-bit
 * with no per-function duplication.
 */
#include <cstring>

#include "common.h"
#include "primitives.h"

namespace X265_NS {
void setupPixelPrimitives_c(EncoderPrimitives &p);
void setupDCTPrimitives_c(EncoderPrimitives &p);
void setupIntraPrimitives_c(EncoderPrimitives &p);

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

#define BPG_PREFIX bpgprim10_
#include "wrapper_impl.h"
