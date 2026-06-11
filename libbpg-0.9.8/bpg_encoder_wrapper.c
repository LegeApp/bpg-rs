#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

// Simple encoder wrapper that calls bpgenc.exe
int bpg_encode_file(const char* input_file, const char* output_file, int quality, int lossless) {
    char cmd[1024];
    if (lossless) {
        snprintf(cmd, sizeof(cmd), "bpgenc.exe -lossless -o %s %s", output_file, input_file);
    } else {
        snprintf(cmd, sizeof(cmd), "bpgenc.exe -q %d -o %s %s", quality, output_file, input_file);
    }
    return system(cmd);
}

// Memory-based encoding (writes to temp file then reads back)
int bpg_encode_memory(const unsigned char* input_data, size_t input_size,
                       unsigned char** output_data, size_t* output_size,
                       int width, int height, int quality, int lossless) {
    // For now, return error - implement file-based encoding first
    return -1;
}
