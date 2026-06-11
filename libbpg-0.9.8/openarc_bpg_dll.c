/*
 * OpenArc BPG Encoder DLL
 * 
 * Provides a Windows DLL interface for BPG encoding with both x265 and JCTVC support.
 * Designed for direct integration into OpenArc applications without subprocess overhead.
 */

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bpg_api.h"

// DLL export macro
#define BPG_DLL_EXPORT __declspec(dllexport)

// Encoder type constants
#define BPG_ENCODER_X265    0
#define BPG_ENCODER_JCTVC   1

// Error handling
static char g_error_msg[256];

/*
 * Batch processing support
 */

typedef struct {
    int count;
    char** input_files;
    char** output_files;
    int* qualities;
    int* encoder_types;
    int* results;
} BPGBatchJob;

// Forward declarations
BPG_DLL_EXPORT void openarc_bpg_destroy_batch_job(BPGBatchJob* job);

// DLL main function
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
        // Initialize library
        break;
    case DLL_THREAD_ATTACH:
        break;
    case DLL_THREAD_DETACH:
        break;
    case DLL_PROCESS_DETACH:
        // Cleanup if needed
        break;
    }
    return TRUE;
}

/*
 * OpenArc-specific API Functions
 */

// Create encoder context with specified type
BPG_DLL_EXPORT BPGEncoderContext* openarc_bpg_create_encoder(int encoder_type) {
    BPGEncoderConfig config;
    bpg_encoder_get_default_config(&config);
    config.encoder_type = encoder_type;
    return bpg_encoder_create_ex(&config);
}

// Simple encode function for file-to-file encoding
BPG_DLL_EXPORT int openarc_bpg_encode_file(
    const char* input_path,
    const char* output_path,
    int quality,
    int encoder_type
) {
    BPGEncoderContext* ctx = openarc_bpg_create_encoder(encoder_type);
    if (!ctx) {
        snprintf(g_error_msg, sizeof(g_error_msg), "Failed to create encoder context");
        return -1;
    }
    
    BPGEncoderConfig config;
    bpg_encoder_get_default_config(&config);
    config.quality = quality;
    config.encoder_type = encoder_type;
    
    int result = bpg_encoder_set_config(ctx, &config);
    if (result != BPG_OK) {
        snprintf(g_error_msg, sizeof(g_error_msg), "Failed to set encoder config: %s", 
                bpg_encoder_get_error(ctx));
        bpg_encoder_destroy(ctx);
        return result;
    }
    
    result = bpg_encode_to_file(ctx, input_path, output_path);
    if (result != BPG_OK) {
        snprintf(g_error_msg, sizeof(g_error_msg), "Encoding failed: %s", 
                bpg_encoder_get_error(ctx));
    }
    
    bpg_encoder_destroy(ctx);
    return result;
}

// Encode from memory to memory (for in-memory processing)
BPG_DLL_EXPORT int openarc_bpg_encode_memory(
    const uint8_t* input_data,
    int width,
    int height,
    int stride,
    int format,
    int quality,
    int encoder_type,
    uint8_t** output_data,
    size_t* output_size
) {
    BPGEncoderContext* ctx = openarc_bpg_create_encoder(encoder_type);
    if (!ctx) {
        snprintf(g_error_msg, sizeof(g_error_msg), "Failed to create encoder context");
        return -1;
    }
    
    BPGEncoderConfig config;
    bpg_encoder_get_default_config(&config);
    config.quality = quality;
    config.encoder_type = encoder_type;
    
    int result = bpg_encoder_set_config(ctx, &config);
    if (result != BPG_OK) {
        snprintf(g_error_msg, sizeof(g_error_msg), "Failed to set encoder config: %s", 
                bpg_encoder_get_error(ctx));
        bpg_encoder_destroy(ctx);
        return result;
    }
    
    result = bpg_encode_from_memory(ctx, input_data, width, height, stride, 
                                  (BPGImageFormat)format, output_data, output_size);
    if (result != BPG_OK) {
        snprintf(g_error_msg, sizeof(g_error_msg), "Encoding failed: %s", 
                bpg_encoder_get_error(ctx));
    }
    
    bpg_encoder_destroy(ctx);
    return result;
}

// Get last error message
BPG_DLL_EXPORT const char* openarc_bpg_get_last_error(void) {
    return g_error_msg;
}

// Get supported encoders
BPG_DLL_EXPORT int openarc_bpg_get_supported_encoders(void) {
    return bpg_get_supported_encoders();
}

// Get library version
BPG_DLL_EXPORT const char* openarc_bpg_get_version(void) {
    return bpg_get_version();
}

// Free memory allocated by BPG functions
BPG_DLL_EXPORT void openarc_bpg_free(void* ptr) {
    bpg_free(ptr);
}

/*
 * Advanced API for fine-grained control
 */

// Create encoder context with custom configuration
BPG_DLL_EXPORT BPGEncoderContext* openarc_bpg_create_encoder_ex(const BPGEncoderConfig* config) {
    return bpg_encoder_create_ex(config);
}

// Set encoder configuration
BPG_DLL_EXPORT int openarc_bpg_set_config(BPGEncoderContext* ctx, const BPGEncoderConfig* config) {
    return bpg_encoder_set_config(ctx, config);
}

// Encode from file to memory
BPG_DLL_EXPORT int openarc_bpg_encode_file_to_memory(
    BPGEncoderContext* ctx,
    const char* input_path,
    uint8_t** output_data,
    size_t* output_size
) {
    return bpg_encode_from_file(ctx, input_path, output_data, output_size);
}

// Get encoder error message
BPG_DLL_EXPORT const char* openarc_bpg_get_encoder_error(BPGEncoderContext* ctx) {
    return bpg_encoder_get_error(ctx);
}

// Destroy encoder context
BPG_DLL_EXPORT void openarc_bpg_destroy_encoder(BPGEncoderContext* ctx) {
    bpg_encoder_destroy(ctx);
}

/*
 * Utility Functions
 */

// Get default configuration
BPG_DLL_EXPORT void openarc_bpg_get_default_config(BPGEncoderConfig* config) {
    bpg_encoder_get_default_config(config);
}

// Check if encoder type is supported
BPG_DLL_EXPORT int openarc_bpg_is_encoder_supported(int encoder_type) {
    int supported = bpg_get_supported_encoders();
    return (supported & (1 << encoder_type)) != 0;
}

// Get encoder name from type
BPG_DLL_EXPORT const char* openarc_bpg_get_encoder_name(int encoder_type) {
    switch (encoder_type) {
    case BPG_ENCODER_X265:
        return "x265";
    case BPG_ENCODER_JCTVC:
        return "JCTVC";
    default:
        return "Unknown";
    }
}

/*
 * Batch processing support
 */

// Create batch job
BPG_DLL_EXPORT BPGBatchJob* openarc_bpg_create_batch_job(int count) {
    BPGBatchJob* job = (BPGBatchJob*)calloc(1, sizeof(BPGBatchJob));
    if (!job) return NULL;
    
    job->count = count;
    job->input_files = (char**)calloc(count, sizeof(char*));
    job->output_files = (char**)calloc(count, sizeof(char*));
    job->qualities = (int*)calloc(count, sizeof(int));
    job->encoder_types = (int*)calloc(count, sizeof(int));
    job->results = (int*)calloc(count, sizeof(int));
    
    if (!job->input_files || !job->output_files || !job->qualities || 
        !job->encoder_types || !job->results) {
        openarc_bpg_destroy_batch_job(job);
        return NULL;
    }
    
    return job;
}

// Set batch job parameters
BPG_DLL_EXPORT int openarc_bpg_set_batch_item(
    BPGBatchJob* job,
    int index,
    const char* input_file,
    const char* output_file,
    int quality,
    int encoder_type
) {
    if (index < 0 || index >= job->count) return -1;
    
    job->input_files[index] = strdup(input_file);
    job->output_files[index] = strdup(output_file);
    job->qualities[index] = quality;
    job->encoder_types[index] = encoder_type;
    
    return 0;
}

// Execute batch job
BPG_DLL_EXPORT int openarc_bpg_execute_batch_job(BPGBatchJob* job) {
    for (int i = 0; i < job->count; i++) {
        job->results[i] = openarc_bpg_encode_file(
            job->input_files[i],
            job->output_files[i],
            job->qualities[i],
            job->encoder_types[i]
        );
    }
    
    // Return 0 if all succeeded, 1 if any failed
    for (int i = 0; i < job->count; i++) {
        if (job->results[i] != BPG_OK) return 1;
    }
    return 0;
}

// Get batch job result
BPG_DLL_EXPORT int openarc_bpg_get_batch_result(BPGBatchJob* job, int index) {
    if (index < 0 || index >= job->count) return -1;
    return job->results[index];
}

// Destroy batch job
BPG_DLL_EXPORT void openarc_bpg_destroy_batch_job(BPGBatchJob* job) {
    if (!job) return;
    
    if (job->input_files) {
        for (int i = 0; i < job->count; i++) {
            free(job->input_files[i]);
        }
        free(job->input_files);
    }
    
    if (job->output_files) {
        for (int i = 0; i < job->count; i++) {
            free(job->output_files[i]);
        }
        free(job->output_files);
    }
    
    free(job->qualities);
    free(job->encoder_types);
    free(job->results);
    free(job);
}
