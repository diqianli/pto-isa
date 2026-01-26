/**
 * PTO Runtime - Ascend A2/A3 Binary Loader Implementation
 *
 * Implements ELF parsing and .text section extraction for AICore kernel binaries.
 * Based on ref_runtime/src/platform/a2a3/host/binary_loader.cpp
 */

#include "a2a3_binary_loader.h"
#include "../a2a3_runtime_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <elf.h>

// =============================================================================
// Internal State
// =============================================================================

#define A2A3_MAX_INCORE_BINARIES 256

static A2A3InCoreBinaryEntry g_incore_binaries[A2A3_MAX_INCORE_BINARIES];
static int g_incore_binary_count = 0;
static bool g_binary_loader_initialized = false;

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Extract function name from .o filename.
 * e.g., "tile_add.o" -> "tile_add"
 *       "/path/to/gemm_tile.o" -> "gemm_tile"
 */
static void extract_func_name_from_path(const char* file_path, char* func_name, size_t max_len) {
    // Find the last '/' in the path
    const char* filename = strrchr(file_path, '/');
    if (filename) {
        filename++;  // Skip the '/'
    } else {
        filename = file_path;
    }
    
    // Copy filename without extension
    strncpy(func_name, filename, max_len - 1);
    func_name[max_len - 1] = '\0';
    
    // Remove .o extension
    char* dot = strrchr(func_name, '.');
    if (dot && strcmp(dot, ".o") == 0) {
        *dot = '\0';
    }
}

/**
 * Check if a file has .o extension.
 */
static bool is_o_file(const char* filename) {
    const char* ext = strrchr(filename, '.');
    return ext && strcmp(ext, ".o") == 0;
}

// =============================================================================
// ELF Binary Loading Implementation
// =============================================================================

uint32_t a2a3_get_file_size(const char* file_path) {
    FILE* file = fopen(file_path, "rb");
    if (!file) {
        fprintf(stderr, "[A2A3 Binary Loader] Error: Cannot open file: %s\n", file_path);
        return 0;
    }
    
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fclose(file);
    
    if (size < 0) {
        fprintf(stderr, "[A2A3 Binary Loader] Error: Invalid file size for: %s\n", file_path);
        return 0;
    }
    
    return (uint32_t)size;
}

int a2a3_load_elf_text_section(const char* bin_path, uint8_t** out_data, size_t* out_size) {
    *out_data = NULL;
    *out_size = 0;
    
    // Step 1: Read entire file
    uint32_t file_size = a2a3_get_file_size(bin_path);
    if (file_size == 0) {
        fprintf(stderr, "[A2A3 Binary Loader] Error: File is empty or cannot be read: %s\n", bin_path);
        return A2A3_ERROR_BINARY_LOAD_FAILED;
    }
    
    uint8_t* buf = (uint8_t*)malloc(file_size);
    if (!buf) {
        fprintf(stderr, "[A2A3 Binary Loader] Error: Failed to allocate buffer for file\n");
        return A2A3_ERROR_MEMORY_ALLOC;
    }
    
    FILE* file = fopen(bin_path, "rb");
    if (!file) {
        fprintf(stderr, "[A2A3 Binary Loader] Error: Cannot open file: %s\n", bin_path);
        free(buf);
        return A2A3_ERROR_BINARY_LOAD_FAILED;
    }
    
    size_t bytes_read = fread(buf, 1, file_size, file);
    fclose(file);
    
    if (bytes_read != file_size) {
        fprintf(stderr, "[A2A3 Binary Loader] Error: Failed to read file: %s\n", bin_path);
        free(buf);
        return A2A3_ERROR_BINARY_LOAD_FAILED;
    }
    
    // Step 2: Parse ELF header
    if (file_size < sizeof(Elf64_Ehdr)) {
        fprintf(stderr, "[A2A3 Binary Loader] Error: File too small for ELF header: %s\n", bin_path);
        free(buf);
        return A2A3_ERROR_BINARY_LOAD_FAILED;
    }
    
    Elf64_Ehdr* elf_header = (Elf64_Ehdr*)buf;
    
    // Verify ELF magic number (0x7F 'E' 'L' 'F')
    if (elf_header->e_ident[EI_MAG0] != ELFMAG0 ||
        elf_header->e_ident[EI_MAG1] != ELFMAG1 ||
        elf_header->e_ident[EI_MAG2] != ELFMAG2 ||
        elf_header->e_ident[EI_MAG3] != ELFMAG3) {
        fprintf(stderr, "[A2A3 Binary Loader] Error: Not a valid ELF file: %s\n", bin_path);
        free(buf);
        return A2A3_ERROR_BINARY_LOAD_FAILED;
    }
    
    // Verify 64-bit ELF
    if (elf_header->e_ident[EI_CLASS] != ELFCLASS64) {
        fprintf(stderr, "[A2A3 Binary Loader] Error: Not a 64-bit ELF file: %s\n", bin_path);
        free(buf);
        return A2A3_ERROR_BINARY_LOAD_FAILED;
    }
    
    // Step 3: Get section headers
    if (elf_header->e_shoff == 0 || elf_header->e_shnum == 0) {
        fprintf(stderr, "[A2A3 Binary Loader] Error: No section headers in ELF file: %s\n", bin_path);
        free(buf);
        return A2A3_ERROR_BINARY_LOAD_FAILED;
    }
    
    Elf64_Shdr* section_headers = (Elf64_Shdr*)(buf + elf_header->e_shoff);
    
    // Get string table for section names
    if (elf_header->e_shstrndx >= elf_header->e_shnum) {
        fprintf(stderr, "[A2A3 Binary Loader] Error: Invalid section string table index: %s\n", bin_path);
        free(buf);
        return A2A3_ERROR_BINARY_LOAD_FAILED;
    }
    
    Elf64_Shdr* shstr_header = &section_headers[elf_header->e_shstrndx];
    char* strtbl = (char*)(buf + shstr_header->sh_offset);
    
    // Step 4: Find and extract .text section
    uint8_t* text_data = NULL;
    size_t text_size = 0;
    
    for (int i = 0; i < elf_header->e_shnum; i++) {
        Elf64_Shdr* section = &section_headers[i];
        char* section_name = strtbl + section->sh_name;
        
        if (strcmp(section_name, ".text") == 0) {
            // Extract .text section binary data
            text_size = section->sh_size;
            text_data = (uint8_t*)malloc(text_size);
            if (!text_data) {
                fprintf(stderr, "[A2A3 Binary Loader] Error: Failed to allocate text section buffer\n");
                free(buf);
                return A2A3_ERROR_MEMORY_ALLOC;
            }
            
            memcpy(text_data, buf + section->sh_offset, text_size);
            
            printf("[A2A3 Binary Loader] Loaded .text section from %s (size: %zu bytes)\n",
                   bin_path, text_size);
            break;
        }
    }
    
    free(buf);
    
    if (!text_data || text_size == 0) {
        fprintf(stderr, "[A2A3 Binary Loader] Error: .text section not found in: %s\n", bin_path);
        return A2A3_ERROR_BINARY_LOAD_FAILED;
    }
    
    *out_data = text_data;
    *out_size = text_size;
    return A2A3_SUCCESS;
}

// =============================================================================
// InCore Binary Registry
// =============================================================================

int a2a3_load_incore_binary_dir(const char* dir_path, bool is_cube) {
    if (!dir_path) {
        fprintf(stderr, "[A2A3 Binary Loader] Error: dir_path is NULL\n");
        return A2A3_ERROR_INVALID_CONFIG;
    }
    
    DIR* dir = opendir(dir_path);
    if (!dir) {
        fprintf(stderr, "[A2A3 Binary Loader] Warning: Cannot open directory: %s\n", dir_path);
        return 0;  // Not an error - directory might not exist
    }
    
    int loaded_count = 0;
    struct dirent* entry;
    
    while ((entry = readdir(dir)) != NULL) {
        // Skip . and ..
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        
        // Check if it's a .o file
        if (!is_o_file(entry->d_name)) {
            continue;
        }
        
        // Build full path
        char full_path[512];
        snprintf(full_path, sizeof(full_path), "%s/%s", dir_path, entry->d_name);
        
        // Check if it's a regular file
        struct stat st;
        if (stat(full_path, &st) != 0 || !S_ISREG(st.st_mode)) {
            continue;
        }
        
        // Load the .o file
        if (a2a3_load_incore_binary(full_path, NULL, is_cube) == 0) {
            loaded_count++;
        }
    }
    
    closedir(dir);
    
    printf("[A2A3 Binary Loader] Loaded %d %s binaries from %s\n",
           loaded_count, is_cube ? "AIC" : "AIV", dir_path);
    
    return loaded_count;
}

int a2a3_load_incore_binary(const char* bin_path, const char* func_name, bool is_cube) {
    if (!bin_path) {
        return A2A3_ERROR_INVALID_CONFIG;
    }
    
    if (g_incore_binary_count >= A2A3_MAX_INCORE_BINARIES) {
        fprintf(stderr, "[A2A3 Binary Loader] Error: Maximum InCore binaries reached (%d)\n",
                A2A3_MAX_INCORE_BINARIES);
        return A2A3_ERROR_MEMORY_ALLOC;
    }
    
    // Extract function name if not provided
    char name_buf[128];
    if (!func_name) {
        extract_func_name_from_path(bin_path, name_buf, sizeof(name_buf));
        func_name = name_buf;
    }
    
    // Check if already loaded
    for (int i = 0; i < g_incore_binary_count; i++) {
        if (g_incore_binaries[i].is_loaded &&
            strcmp(g_incore_binaries[i].func_name, func_name) == 0) {
            printf("[A2A3 Binary Loader] Binary '%s' already loaded, skipping\n", func_name);
            return 0;
        }
    }
    
    // Load ELF .text section
    uint8_t* binary_data = NULL;
    size_t binary_size = 0;
    
    int rc = a2a3_load_elf_text_section(bin_path, &binary_data, &binary_size);
    if (rc != A2A3_SUCCESS) {
        return rc;
    }
    
    // Register the binary
    A2A3InCoreBinaryEntry* entry = &g_incore_binaries[g_incore_binary_count];
    strncpy(entry->func_name, func_name, sizeof(entry->func_name) - 1);
    entry->func_name[sizeof(entry->func_name) - 1] = '\0';
    entry->binary_data = binary_data;
    entry->binary_size = binary_size;
    entry->device_addr = 0;  // Will be set when loaded to device
    entry->is_cube = is_cube;
    entry->is_loaded = true;
    g_incore_binary_count++;
    
    printf("[A2A3 Binary Loader] Loaded InCore binary: %s -> %s (%zu bytes) [%s]\n",
           bin_path, func_name, binary_size, is_cube ? "AIC" : "AIV");
    
    return A2A3_SUCCESS;
}

A2A3InCoreBinaryEntry* a2a3_lookup_incore_binary(const char* func_name) {
    if (!func_name) return NULL;
    
    for (int i = 0; i < g_incore_binary_count; i++) {
        if (g_incore_binaries[i].is_loaded &&
            strcmp(g_incore_binaries[i].func_name, func_name) == 0) {
            return &g_incore_binaries[i];
        }
    }
    
    return NULL;
}

int a2a3_get_incore_binary_count(void) {
    return g_incore_binary_count;
}

void a2a3_unload_all_incore_binaries(void) {
    for (int i = 0; i < g_incore_binary_count; i++) {
        if (g_incore_binaries[i].is_loaded && g_incore_binaries[i].binary_data) {
            free(g_incore_binaries[i].binary_data);
        }
#ifdef CANN_SDK_AVAILABLE
        // Free device memory if allocated
        if (g_incore_binaries[i].device_addr != 0) {
            aclrtFree((void*)g_incore_binaries[i].device_addr);
        }
#endif
        g_incore_binaries[i].is_loaded = false;
        g_incore_binaries[i].binary_data = NULL;
        g_incore_binaries[i].binary_size = 0;
        g_incore_binaries[i].device_addr = 0;
    }
    g_incore_binary_count = 0;
}

// =============================================================================
// Device Memory Operations
// =============================================================================

#ifdef CANN_SDK_AVAILABLE
#include <acl/acl.h>
#else
// Stub definitions for compilation without CANN SDK
static inline int aclrtFree(void* ptr) { (void)ptr; return 0; }
static inline int aclrtMalloc(void** ptr, size_t size, int flag) { 
    *ptr = malloc(size); 
    (void)flag;
    return (*ptr != NULL) ? 0 : -1; 
}
static inline int aclrtMemcpy(void* dst, size_t dstMax, const void* src, size_t count, int kind) {
    (void)dstMax; (void)kind;
    memcpy(dst, src, count);
    return 0;
}
#define ACL_MEM_MALLOC_HUGE_FIRST 0
#define ACL_MEMCPY_HOST_TO_DEVICE 1
#define ACL_SUCCESS 0
#endif

int a2a3_copy_incore_binaries_to_device(void) {
    int copied = 0;
    
    for (int i = 0; i < g_incore_binary_count; i++) {
        A2A3InCoreBinaryEntry* entry = &g_incore_binaries[i];
        
        if (!entry->is_loaded || !entry->binary_data || entry->binary_size == 0) {
            continue;
        }
        
        // Skip if already copied
        if (entry->device_addr != 0) {
            copied++;
            continue;
        }
        
#ifdef CANN_SDK_AVAILABLE
        // Allocate device GM memory
        void* dev_ptr = NULL;
        aclError rc = aclrtMalloc(&dev_ptr, entry->binary_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (rc != ACL_SUCCESS) {
            fprintf(stderr, "[A2A3 Binary Loader] Error: Failed to allocate device memory for %s: %d\n",
                    entry->func_name, rc);
            continue;
        }
        
        // Copy binary to device
        rc = aclrtMemcpy(dev_ptr, entry->binary_size, entry->binary_data, 
                         entry->binary_size, ACL_MEMCPY_HOST_TO_DEVICE);
        if (rc != ACL_SUCCESS) {
            fprintf(stderr, "[A2A3 Binary Loader] Error: Failed to copy %s to device: %d\n",
                    entry->func_name, rc);
            aclrtFree(dev_ptr);
            continue;
        }
        
        entry->device_addr = (uint64_t)dev_ptr;
        printf("[A2A3 Binary Loader] Copied %s to device: 0x%lx (%zu bytes)\n",
               entry->func_name, entry->device_addr, entry->binary_size);
        copied++;
#else
        // Stub mode: use host address as device address (for testing)
        entry->device_addr = (uint64_t)entry->binary_data;
        printf("[A2A3 Binary Loader] Stub mode: %s -> 0x%lx (%zu bytes)\n",
               entry->func_name, entry->device_addr, entry->binary_size);
        copied++;
#endif
    }
    
    printf("[A2A3 Binary Loader] Copied %d binaries to device\n", copied);
    return copied;
}

uint64_t a2a3_get_incore_device_addr(const char* func_name) {
    A2A3InCoreBinaryEntry* entry = a2a3_lookup_incore_binary(func_name);
    if (entry && entry->device_addr != 0) {
        return entry->device_addr;
    }
    return 0;
}

// =============================================================================
// Initialization and Cleanup
// =============================================================================

void a2a3_binary_loader_init(void) {
    if (g_binary_loader_initialized) return;
    
    memset(g_incore_binaries, 0, sizeof(g_incore_binaries));
    g_incore_binary_count = 0;
    g_binary_loader_initialized = true;
    
    printf("[A2A3 Binary Loader] Initialized\n");
}

void a2a3_binary_loader_cleanup(void) {
    if (!g_binary_loader_initialized) return;
    
    a2a3_unload_all_incore_binaries();
    g_binary_loader_initialized = false;
    
    printf("[A2A3 Binary Loader] Cleanup complete\n");
}
