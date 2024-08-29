#if _WIN32
#    include <windows.h>
#endif

#include "NRIFramework.h"

struct DebugAllocator {
    std::atomic_uint64_t allocationNum = 0;
    std::atomic_size_t allocatedSize = 0;
};

struct DebugAllocatorHeader {
    size_t size;
    uint32_t alignment;
    uint32_t offset;
};

inline void ReportAllocatorError(const char* message) {
#if _WIN32
    OutputDebugStringA(message);
#endif
    std::cout << message;
    std::abort();
}

inline DebugAllocatorHeader* GetAllocationHeader(void* memory) {
    return (DebugAllocatorHeader*)memory - 1;
}

static void* DebugAlignedMalloc(void* userArg, size_t size, size_t alignment) {
    DebugAllocator* allocator = (DebugAllocator*)userArg;

    if (alignment == 0)
        ReportAllocatorError("DebugAlignedMalloc() failed: alignment can't be 0.\n");

    const size_t alignedHeaderSize = helper::Align(sizeof(DebugAllocatorHeader), alignment);
    const size_t allocationSize = size + alignment - 1 + alignedHeaderSize;

    uint8_t* memory = (uint8_t*)malloc(allocationSize);

    if (memory == nullptr)
        return nullptr;

    uint8_t* alignedMemory = helper::Align(memory, alignment) + alignedHeaderSize;

    DebugAllocatorHeader* header = GetAllocationHeader(alignedMemory);
    *header = {};
    header->size = allocationSize;
    header->alignment = (uint32_t)alignment;
    header->offset = (uint32_t)(alignedMemory - memory);

    allocator->allocatedSize.fetch_add(allocationSize, std::memory_order_relaxed);
    allocator->allocationNum.fetch_add(1, std::memory_order_relaxed);

    return alignedMemory;
}

static void* DebugAlignedRealloc(void* userArg, void* memory, size_t size, size_t alignment) {
    DebugAllocator* allocator = (DebugAllocator*)userArg;

    if (alignment == 0)
        ReportAllocatorError("DebugAlignedRealloc() failed: alignment can't be 0.\n");

    if (memory == nullptr)
        return DebugAlignedMalloc(userArg, size, alignment);

    const DebugAllocatorHeader prevHeader = *GetAllocationHeader(memory);

    if (prevHeader.alignment != alignment)
        ReportAllocatorError("DebugAlignedRealloc() failed: memory alignment mismatch.\n");

    const size_t alignedHeaderSize = helper::Align(sizeof(DebugAllocatorHeader), alignment);
    const size_t allocationSize = size + alignment - 1 + alignedHeaderSize;

    uint8_t* prevMemoryBegin = (uint8_t*)memory - prevHeader.offset;

    uint8_t* newMemory = (uint8_t*)realloc(prevMemoryBegin, allocationSize);

    if (newMemory == nullptr)
        return nullptr;

    uint8_t* alignedMemory = helper::Align(newMemory, alignment) + alignedHeaderSize;

    allocator->allocatedSize.fetch_add(allocationSize - prevHeader.size, std::memory_order_relaxed);

    DebugAllocatorHeader* newHeader = GetAllocationHeader(alignedMemory);
    *newHeader = {};
    newHeader->size = allocationSize;
    newHeader->alignment = (uint32_t)alignment;
    newHeader->offset = (uint32_t)(alignedMemory - newMemory);

    return alignedMemory;
}

static void DebugAlignedFree(void* userArg, void* memory) {
    if (memory == nullptr)
        return;

    const DebugAllocatorHeader* header = GetAllocationHeader(memory);

    DebugAllocator* allocator = (DebugAllocator*)userArg;
    const size_t allocatedSize = allocator->allocatedSize.fetch_sub(header->size, std::memory_order_relaxed);
    const size_t allocationNum = allocator->allocationNum.fetch_sub(1, std::memory_order_relaxed);

    if (allocatedSize < header->size)
        ReportAllocatorError("DebugAlignedFree() failed: invalid allocated size.\n");
    if (allocationNum == 0)
        ReportAllocatorError("DebugAlignedFree() failed: invalid allocation number.\n");

    free((uint8_t*)memory - header->offset);
}

void CreateDebugAllocator(nri::AllocationCallbacks& allocationCallbacks) {
    allocationCallbacks = {};
    allocationCallbacks.userArg = new DebugAllocator();
    allocationCallbacks.Allocate = DebugAlignedMalloc;
    allocationCallbacks.Reallocate = DebugAlignedRealloc;
    allocationCallbacks.Free = DebugAlignedFree;
}

void DestroyDebugAllocator(nri::AllocationCallbacks& allocationCallbacks) {
    DebugAllocator* debugAllocator = (DebugAllocator*)allocationCallbacks.userArg;

    if (debugAllocator->allocatedSize.load(std::memory_order_relaxed) != 0)
        ReportAllocatorError("DestroyDebugAllocator() failed: allocatedSize is not 0.\n");
    if (debugAllocator->allocationNum.load(std::memory_order_relaxed) != 0)
        ReportAllocatorError("DestroyDebugAllocator() failed: allocationNum is not 0.\n");

    delete debugAllocator;
}