/*
 * BPG Native Library API Implementation
 */

#include "bpg_api.h"
#include "libbpg.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define BPG_VERSION "0.9.8-native"

/* Encoder context structure */
struct BPGEncoderContext {
    BPGEncoderConfig config;
    char error_msg[256];
    int has_error;
};

/* Get default configuration */
void bpg_encoder_get_default_config(BPGEncoderConfig* config) {
    if (!config) return;
    
    config->quality = 28;
    config->bit_depth = 8;
    config->lossless = 0;
    config->chroma_format = 1;  /* 4:2:0 */
    config->encoder_type = 0;   /* x265 */
    config->compress_level = 8;
    config->color_space = 3;    /* YCbCr BT.709 (better for HEIC/HEIF sources) */
    config->limited_range = 0;  /* full range */
}

/* Create encoder with default configuration */
BPGEncoderContext* bpg_encoder_create(void) {
    BPGEncoderContext* ctx = (BPGEncoderContext*)calloc(1, sizeof(BPGEncoderContext));
    if (!ctx) return NULL;
    
    bpg_encoder_get_default_config(&ctx->config);
    ctx->has_error = 0;
    ctx->error_msg[0] = '\0';
    
    return ctx;
}

/* Create encoder with custom configuration */
BPGEncoderContext* bpg_encoder_create_ex(const BPGEncoderConfig* config) {
    BPGEncoderContext* ctx = bpg_encoder_create();
    if (!ctx) return NULL;
    
    if (config) {
        memcpy(&ctx->config, config, sizeof(BPGEncoderConfig));
    }
    
    return ctx;
}

/* Set encoder configuration */
int bpg_encoder_set_config(BPGEncoderContext* ctx, const BPGEncoderConfig* config) {
    if (!ctx || !config) return BPG_ERROR_INVALID_PARAM;
    
    /* Validate configuration */
    if (config->quality < 0 || config->quality > 51) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg), 
                 "Invalid quality: %d (must be 0-51)", config->quality);
        ctx->has_error = 1;
        return BPG_ERROR_INVALID_PARAM;
    }
    
    if (config->bit_depth != 8 && config->bit_depth != 10 && config->bit_depth != 12) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Invalid bit depth: %d (must be 8, 10, or 12)", config->bit_depth);
        ctx->has_error = 1;
        return BPG_ERROR_INVALID_PARAM;
    }

    if (config->limited_range != 0 && config->limited_range != 1) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Invalid limited_range: %d (must be 0 or 1)", config->limited_range);
        ctx->has_error = 1;
        return BPG_ERROR_INVALID_PARAM;
    }
    
    memcpy(&ctx->config, config, sizeof(BPGEncoderConfig));
    return BPG_OK;
}

/* Set error message */
static void set_error(BPGEncoderContext* ctx, const char* msg) {
    if (!ctx) return;
    snprintf(ctx->error_msg, sizeof(ctx->error_msg), "%s", msg);
    ctx->has_error = 1;
}

/* Get last error message */
const char* bpg_encoder_get_error(BPGEncoderContext* ctx) {
    if (!ctx || !ctx->has_error) return "No error";
    return ctx->error_msg;
}

/* Destroy encoder context */
void bpg_encoder_destroy(BPGEncoderContext* ctx) {
    if (ctx) {
        free(ctx);
    }
}

/* Free memory allocated by BPG API */
void bpg_free(void* ptr) {
    if (ptr) {
        free(ptr);
    }
}

/* Get library version */
const char* bpg_get_version(void) {
    return BPG_VERSION;
}

/* Get supported encoders */
int bpg_get_supported_encoders(void) {
    int encoders = 0x01;  /* x265 always supported */
#ifdef USE_JCTVC
    encoders |= 0x02;     /* JCTVC if compiled with it */
#endif
    return encoders;
}

/*
 * Encoding implementation - uses bpgenc's x265 encoder directly in-process.
 */

/* Extern declaration for the library encoding function in bpgenc.c */
extern int bpgenc_encode_from_memory_buffer(
    const uint8_t *input_data,
    int width, int height, int stride,
    int input_format,
    int quality, int bit_depth, int lossless, int chroma_format,
    int compress_level, int encoder_type, int color_space, int limited_range,
    uint8_t **output_data, size_t *output_size);
extern int bpgenc_encode_from_planar_u8_buffer(
    const uint8_t *y_plane, int y_stride,
    const uint8_t *cb_plane, int cb_stride,
    const uint8_t *cr_plane, int cr_stride,
    int width, int height, int input_format,
    int quality, int bit_depth, int lossless, int chroma_format,
    int compress_level, int encoder_type, int color_space, int limited_range,
    uint8_t **output_data, size_t *output_size);
extern int bpgenc_encode_from_planar_u16_buffer(
    const uint16_t *y_plane, int y_stride,
    const uint16_t *cb_plane, int cb_stride,
    const uint16_t *cr_plane, int cr_stride,
    int width, int height, int input_format,
    int quality, int bit_depth, int lossless, int chroma_format,
    int compress_level, int encoder_type, int color_space, int limited_range,
    uint8_t **output_data, size_t *output_size);

/* Encode from file - loads the file and encodes in memory */
int bpg_encode_from_file(
    BPGEncoderContext* ctx,
    const char* input_path,
    uint8_t** output_data,
    size_t* output_size
) {
    if (!ctx || !input_path || !output_data || !output_size) {
        return BPG_ERROR_INVALID_PARAM;
    }

    /* Read the file into memory, determine format, then encode */
    FILE* f = fopen(input_path, "rb");
    if (!f) {
        set_error(ctx, "Cannot open input file");
        return BPG_ERROR_FILE_IO;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size <= 0) {
        fclose(f);
        set_error(ctx, "Empty input file");
        return BPG_ERROR_FILE_IO;
    }

    /* For file-based encoding, use encode_to_file then read result */
    /* This is a fallback - prefer encode_from_memory for decoded pixel data */
    fclose(f);
    set_error(ctx, "Use bpg_encode_from_memory with decoded pixel data instead");
    return BPG_ERROR_UNSUPPORTED_FORMAT;
}

/* Encode from memory buffer - direct in-process encoding via x265 */
int bpg_encode_from_memory(
    BPGEncoderContext* ctx,
    const uint8_t* input_data,
    int width,
    int height,
    int stride,
    BPGImageFormat format,
    uint8_t** output_data,
    size_t* output_size
) {
    int ret;

    if (!ctx || !input_data || !output_data || !output_size) {
        return BPG_ERROR_INVALID_PARAM;
    }

    if (width <= 0 || height <= 0) {
        set_error(ctx, "Invalid image dimensions");
        return BPG_ERROR_INVALID_PARAM;
    }

    ret = bpgenc_encode_from_memory_buffer(
        input_data, width, height, stride,
        (int)format,
        ctx->config.quality,
        ctx->config.bit_depth,
        ctx->config.lossless,
        ctx->config.chroma_format,
        ctx->config.compress_level,
        ctx->config.encoder_type,
        ctx->config.color_space,
        ctx->config.limited_range,
        output_data, output_size);

    if (ret < 0) {
        switch (ret) {
        case -1: set_error(ctx, "Invalid parameters"); return BPG_ERROR_INVALID_PARAM;
        case -2: set_error(ctx, "Out of memory"); return BPG_ERROR_OUT_OF_MEMORY;
        case -3: set_error(ctx, "Unsupported input format"); return BPG_ERROR_UNSUPPORTED_FORMAT;
        case -4: set_error(ctx, "Encoder initialization failed"); return BPG_ERROR_ENCODE_FAILED;
        case -5: set_error(ctx, "x265 encoding failed"); return BPG_ERROR_ENCODE_FAILED;
        default: set_error(ctx, "Encoding failed"); return BPG_ERROR_ENCODE_FAILED;
        }
    }

    return BPG_OK;
}

int bpg_encode_from_planar_u8(
    BPGEncoderContext* ctx,
    const uint8_t* y_plane,
    int y_stride,
    const uint8_t* cb_plane,
    int cb_stride,
    const uint8_t* cr_plane,
    int cr_stride,
    int width,
    int height,
    BPGImageFormat format,
    uint8_t** output_data,
    size_t* output_size
) {
    int ret;

    if (!ctx || !y_plane || !output_data || !output_size) {
        return BPG_ERROR_INVALID_PARAM;
    }

    if (width <= 0 || height <= 0) {
        set_error(ctx, "Invalid image dimensions");
        return BPG_ERROR_INVALID_PARAM;
    }

    ret = bpgenc_encode_from_planar_u8_buffer(
        y_plane, y_stride,
        cb_plane, cb_stride,
        cr_plane, cr_stride,
        width, height, (int)format,
        ctx->config.quality,
        ctx->config.bit_depth,
        ctx->config.lossless,
        ctx->config.chroma_format,
        ctx->config.compress_level,
        ctx->config.encoder_type,
        ctx->config.color_space,
        ctx->config.limited_range,
        output_data, output_size);

    if (ret < 0) {
        switch (ret) {
        case -1: set_error(ctx, "Invalid planar parameters"); return BPG_ERROR_INVALID_PARAM;
        case -2: set_error(ctx, "Out of memory"); return BPG_ERROR_OUT_OF_MEMORY;
        case -3: set_error(ctx, "Unsupported planar input format"); return BPG_ERROR_UNSUPPORTED_FORMAT;
        case -4: set_error(ctx, "Encoder initialization failed"); return BPG_ERROR_ENCODE_FAILED;
        case -5: set_error(ctx, "x265 encoding failed"); return BPG_ERROR_ENCODE_FAILED;
        default: set_error(ctx, "Planar encoding failed"); return BPG_ERROR_ENCODE_FAILED;
        }
    }

    return BPG_OK;
}

int bpg_encode_from_planar_u16(
    BPGEncoderContext* ctx,
    const uint16_t* y_plane,
    int y_stride,
    const uint16_t* cb_plane,
    int cb_stride,
    const uint16_t* cr_plane,
    int cr_stride,
    int width,
    int height,
    BPGImageFormat format,
    uint8_t** output_data,
    size_t* output_size
) {
    int ret;

    if (!ctx || !y_plane || !output_data || !output_size) {
        return BPG_ERROR_INVALID_PARAM;
    }

    if (width <= 0 || height <= 0) {
        set_error(ctx, "Invalid image dimensions");
        return BPG_ERROR_INVALID_PARAM;
    }

    ret = bpgenc_encode_from_planar_u16_buffer(
        y_plane, y_stride,
        cb_plane, cb_stride,
        cr_plane, cr_stride,
        width, height, (int)format,
        ctx->config.quality,
        ctx->config.bit_depth,
        ctx->config.lossless,
        ctx->config.chroma_format,
        ctx->config.compress_level,
        ctx->config.encoder_type,
        ctx->config.color_space,
        ctx->config.limited_range,
        output_data, output_size);

    if (ret < 0) {
        switch (ret) {
        case -1: set_error(ctx, "Invalid planar parameters"); return BPG_ERROR_INVALID_PARAM;
        case -2: set_error(ctx, "Out of memory"); return BPG_ERROR_OUT_OF_MEMORY;
        case -3: set_error(ctx, "Unsupported planar input format"); return BPG_ERROR_UNSUPPORTED_FORMAT;
        case -4: set_error(ctx, "Encoder initialization failed"); return BPG_ERROR_ENCODE_FAILED;
        case -5: set_error(ctx, "x265 encoding failed"); return BPG_ERROR_ENCODE_FAILED;
        default: set_error(ctx, "Planar encoding failed"); return BPG_ERROR_ENCODE_FAILED;
        }
    }

    return BPG_OK;
}

/* Encode to file - encodes in memory then writes to disk */
int bpg_encode_to_file(
    BPGEncoderContext* ctx,
    const char* input_path,
    const char* output_path
) {
    if (!ctx || !input_path || !output_path) {
        return BPG_ERROR_INVALID_PARAM;
    }

    /* This function is deprecated in favor of encode_from_memory + file write.
     * Keep as a fallback using the encode pipeline. */
    set_error(ctx, "Use bpg_encode_from_memory and write result to file");
    return BPG_ERROR_UNSUPPORTED_FORMAT;
}

/*
 * Decoder implementation (using existing libbpg)
 */

int bpg_decode_file(
    const char* input_path,
    uint8_t** output_data,
    int* width,
    int* height,
    BPGImageFormat* format
) {
    if (!input_path || !output_data || !width || !height) {
        return BPG_ERROR_INVALID_PARAM;
    }
    
    /* Use existing libbpg decoder */
    BPGDecoderContext* img = bpg_decoder_open();
    if (!img) {
        return BPG_ERROR_OUT_OF_MEMORY;
    }
    
    /* Load BPG file */
    FILE* f = fopen(input_path, "rb");
    if (!f) {
        bpg_decoder_close(img);
        return BPG_ERROR_FILE_IO;
    }
    
    /* Get file size */
    fseek(f, 0, SEEK_END);
    size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    /* Read file */
    uint8_t* buf = (uint8_t*)malloc(file_size);
    if (!buf) {
        fclose(f);
        bpg_decoder_close(img);
        return BPG_ERROR_OUT_OF_MEMORY;
    }
    
    if (fread(buf, 1, file_size, f) != file_size) {
        free(buf);
        fclose(f);
        bpg_decoder_close(img);
        return BPG_ERROR_FILE_IO;
    }
    fclose(f);
    
    /* Decode */
    if (bpg_decoder_decode(img, buf, file_size) < 0) {
        free(buf);
        bpg_decoder_close(img);
        return BPG_ERROR_DECODE_FAILED;
    }
    free(buf);
    
    /* Get image info */
    BPGImageInfo info;
    bpg_decoder_get_info(img, &info);
    *width = info.width;
    *height = info.height;
    
    /* Allocate output buffer (RGBA32) */
    size_t output_size = info.width * info.height * 4;
    *output_data = (uint8_t*)malloc(output_size);
    if (!*output_data) {
        bpg_decoder_close(img);
        return BPG_ERROR_OUT_OF_MEMORY;
    }
    
    /* Start decoding */
    if (bpg_decoder_start(img, BPG_OUTPUT_FORMAT_RGBA32) < 0) {
        free(*output_data);
        *output_data = NULL;
        bpg_decoder_close(img);
        return BPG_ERROR_DECODE_FAILED;
    }
    
    /* Decode lines */
    for (int y = 0; y < info.height; y++) {
        if (bpg_decoder_get_line(img, *output_data + y * info.width * 4) < 0) {
            free(*output_data);
            *output_data = NULL;
            bpg_decoder_close(img);
            return BPG_ERROR_DECODE_FAILED;
        }
    }
    
    if (format) {
        *format = BPG_INPUT_FORMAT_RGBA32;
    }
    
    bpg_decoder_close(img);
    return BPG_OK;
}
