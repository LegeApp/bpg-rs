/* CABAC entropy-cost tables referenced by `common/dct.cpp`'s
 * `costCoeffNxN_c`/`costC1C2Flag_c` (residual-coding RDO cost estimation,
 * part of `setupDCTPrimitives_c`'s translation unit but not part of the
 * DCT/IDCT primitives this crate currently exposes).
 *
 * The real definitions live in `x265_4.1/source/encoder/entropy.cpp`
 * (`g_entropyBits`, `g_nextState`, `PFX(entropyStateBits)`), which would
 * pull in the entire CABAC entropy-coder translation unit just to satisfy
 * the linker for functions this crate never calls. Copied verbatim here --
 * these are precomputed constant tables (not implementation logic), so
 * copying them doesn't fork any behavior.
 */
#include <stdint.h>
#include "common.h"
#include "primitives.h"

namespace X265_NS {

extern const uint32_t g_entropyBits[128] =
{
    // Corrected table, most notably for last state
    0x07b23, 0x085f9, 0x074a0, 0x08cbc, 0x06ee4, 0x09354, 0x067f4, 0x09c1b, 0x060b0, 0x0a62a, 0x05a9c, 0x0af5b, 0x0548d, 0x0b955, 0x04f56, 0x0c2a9,
    0x04a87, 0x0cbf7, 0x045d6, 0x0d5c3, 0x04144, 0x0e01b, 0x03d88, 0x0e937, 0x039e0, 0x0f2cd, 0x03663, 0x0fc9e, 0x03347, 0x10600, 0x03050, 0x10f95,
    0x02d4d, 0x11a02, 0x02ad3, 0x12333, 0x0286e, 0x12cad, 0x02604, 0x136df, 0x02425, 0x13f48, 0x021f4, 0x149c4, 0x0203e, 0x1527b, 0x01e4d, 0x15d00,
    0x01c99, 0x166de, 0x01b18, 0x17017, 0x019a5, 0x17988, 0x01841, 0x18327, 0x016df, 0x18d50, 0x015d9, 0x19547, 0x0147c, 0x1a083, 0x0138e, 0x1a8a3,
    0x01251, 0x1b418, 0x01166, 0x1bd27, 0x01068, 0x1c77b, 0x00f7f, 0x1d18e, 0x00eda, 0x1d91a, 0x00e19, 0x1e254, 0x00d4f, 0x1ec9a, 0x00c90, 0x1f6e0,
    0x00c01, 0x1fef8, 0x00b5f, 0x208b1, 0x00ab6, 0x21362, 0x00a15, 0x21e46, 0x00988, 0x2285d, 0x00934, 0x22ea8, 0x008a8, 0x239b2, 0x0081d, 0x24577,
    0x007c9, 0x24ce6, 0x00763, 0x25663, 0x00710, 0x25e8f, 0x006a0, 0x26a26, 0x00672, 0x26f23, 0x005e8, 0x27ef8, 0x005ba, 0x284b5, 0x0055e, 0x29057,
    0x0050c, 0x29bab, 0x004c1, 0x2a674, 0x004a7, 0x2aa5e, 0x0046f, 0x2b32f, 0x0041f, 0x2c0ad, 0x003e7, 0x2ca8d, 0x003ba, 0x2d323, 0x0010c, 0x3bfbb
};

extern const uint8_t g_nextState[128][2] =
{
    { 2, 1 }, { 0, 3 }, { 4, 0 }, { 1, 5 }, { 6, 2 }, { 3, 7 }, { 8, 4 }, { 5, 9 },
    { 10, 4 }, { 5, 11 }, { 12, 8 }, { 9, 13 }, { 14, 8 }, { 9, 15 }, { 16, 10 }, { 11, 17 },
    { 18, 12 }, { 13, 19 }, { 20, 14 }, { 15, 21 }, { 22, 16 }, { 17, 23 }, { 24, 18 }, { 19, 25 },
    { 26, 18 }, { 19, 27 }, { 28, 22 }, { 23, 29 }, { 30, 22 }, { 23, 31 }, { 32, 24 }, { 25, 33 },
    { 34, 26 }, { 27, 35 }, { 36, 26 }, { 27, 37 }, { 38, 30 }, { 31, 39 }, { 40, 30 }, { 31, 41 },
    { 42, 32 }, { 33, 43 }, { 44, 32 }, { 33, 45 }, { 46, 36 }, { 37, 47 }, { 48, 36 }, { 37, 49 },
    { 50, 38 }, { 39, 51 }, { 52, 38 }, { 39, 53 }, { 54, 42 }, { 43, 55 }, { 56, 42 }, { 43, 57 },
    { 58, 44 }, { 45, 59 }, { 60, 44 }, { 45, 61 }, { 62, 46 }, { 47, 63 }, { 64, 48 }, { 49, 65 },
    { 66, 48 }, { 49, 67 }, { 68, 50 }, { 51, 69 }, { 70, 52 }, { 53, 71 }, { 72, 52 }, { 53, 73 },
    { 74, 54 }, { 55, 75 }, { 76, 54 }, { 55, 77 }, { 78, 56 }, { 57, 79 }, { 80, 58 }, { 59, 81 },
    { 82, 58 }, { 59, 83 }, { 84, 60 }, { 61, 85 }, { 86, 60 }, { 61, 87 }, { 88, 60 }, { 61, 89 },
    { 90, 62 }, { 63, 91 }, { 92, 64 }, { 65, 93 }, { 94, 64 }, { 65, 95 }, { 96, 66 }, { 67, 97 },
    { 98, 66 }, { 67, 99 }, { 100, 66 }, { 67, 101 }, { 102, 68 }, { 69, 103 }, { 104, 68 }, { 69, 105 },
    { 106, 70 }, { 71, 107 }, { 108, 70 }, { 71, 109 }, { 110, 70 }, { 71, 111 }, { 112, 72 }, { 73, 113 },
    { 114, 72 }, { 73, 115 }, { 116, 72 }, { 73, 117 }, { 118, 74 }, { 75, 119 }, { 120, 74 }, { 75, 121 },
    { 122, 74 }, { 75, 123 }, { 124, 76 }, { 77, 125 }, { 124, 76 }, { 77, 125 }, { 126, 126 }, { 127, 127 }
};

} // namespace X265_NS

// [8 24] --> [stateMPS BitCost], [stateLPS BitCost]
extern "C" const uint32_t PFX(entropyStateBits)[128] =
{
    // Corrected table, most notably for last state
    0x02007B23, 0x000085F9, 0x040074A0, 0x00008CBC, 0x06006EE4, 0x02009354, 0x080067F4, 0x04009C1B,
    0x0A0060B0, 0x0400A62A, 0x0C005A9C, 0x0800AF5B, 0x0E00548D, 0x0800B955, 0x10004F56, 0x0A00C2A9,
    0x12004A87, 0x0C00CBF7, 0x140045D6, 0x0E00D5C3, 0x16004144, 0x1000E01B, 0x18003D88, 0x1200E937,
    0x1A0039E0, 0x1200F2CD, 0x1C003663, 0x1600FC9E, 0x1E003347, 0x16010600, 0x20003050, 0x18010F95,
    0x22002D4D, 0x1A011A02, 0x24002AD3, 0x1A012333, 0x2600286E, 0x1E012CAD, 0x28002604, 0x1E0136DF,
    0x2A002425, 0x20013F48, 0x2C0021F4, 0x200149C4, 0x2E00203E, 0x2401527B, 0x30001E4D, 0x24015D00,
    0x32001C99, 0x260166DE, 0x34001B18, 0x26017017, 0x360019A5, 0x2A017988, 0x38001841, 0x2A018327,
    0x3A0016DF, 0x2C018D50, 0x3C0015D9, 0x2C019547, 0x3E00147C, 0x2E01A083, 0x4000138E, 0x3001A8A3,
    0x42001251, 0x3001B418, 0x44001166, 0x3201BD27, 0x46001068, 0x3401C77B, 0x48000F7F, 0x3401D18E,
    0x4A000EDA, 0x3601D91A, 0x4C000E19, 0x3601E254, 0x4E000D4F, 0x3801EC9A, 0x50000C90, 0x3A01F6E0,
    0x52000C01, 0x3A01FEF8, 0x54000B5F, 0x3C0208B1, 0x56000AB6, 0x3C021362, 0x58000A15, 0x3C021E46,
    0x5A000988, 0x3E02285D, 0x5C000934, 0x40022EA8, 0x5E0008A8, 0x400239B2, 0x6000081D, 0x42024577,
    0x620007C9, 0x42024CE6, 0x64000763, 0x42025663, 0x66000710, 0x44025E8F, 0x680006A0, 0x44026A26,
    0x6A000672, 0x46026F23, 0x6C0005E8, 0x46027EF8, 0x6E0005BA, 0x460284B5, 0x7000055E, 0x48029057,
    0x7200050C, 0x48029BAB, 0x740004C1, 0x4802A674, 0x760004A7, 0x4A02AA5E, 0x7800046F, 0x4A02B32F,
    0x7A00041F, 0x4A02C0AD, 0x7C0003E7, 0x4C02CA8D, 0x7C0003BA, 0x4C02D323, 0x7E00010C, 0x7E03BFBB,
};
