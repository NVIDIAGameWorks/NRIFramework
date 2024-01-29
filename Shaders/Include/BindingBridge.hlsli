// © 2021 NVIDIA Corporation


#if( defined( __cplusplus ) )
    #define NRI_RESOURCE( resourceType, resourceName, regName, bindingIndex, setIndex ) \
        struct resourceName
#elif( defined( COMPILER_DXC ) )
    #if( defined( VULKAN ) )
        #define NRI_RESOURCE( resourceType, resourceName, regName, bindingIndex, setIndex ) \
            resourceType resourceName : register( regName ## bindingIndex, space ## setIndex )

        #define NRI_PUSH_CONSTANTS( structName, constantBufferName, bindingIndex ) \
            [[vk::push_constant]] structName constantBufferName
    #else
        #define NRI_RESOURCE( resourceType, resourceName, regName, bindingIndex, setIndex ) \
            resourceType resourceName : register( regName ## bindingIndex, space ## setIndex )

        #define NRI_PUSH_CONSTANTS( structName, constantBufferName, bindingIndex ) \
            ConstantBuffer<structName> constantBufferName : register( b ## bindingIndex, space0 )
    #endif
#else
    #define NRI_RESOURCE( resourceType, resourceName, regName, bindingIndex, setIndex ) \
        resourceType resourceName : register( regName ## bindingIndex )

    #define NRI_PUSH_CONSTANTS( structName, constantBufferName, bindingIndex ) \
        cbuffer structName ## _ ## constantBufferName : register( b ## bindingIndex ) \
        { \
            structName constantBufferName; \
        }
#endif

// Printf
#ifdef __hlsl_dx_compiler
    #if (!defined(VULKAN))
        #define printf(...)
    #endif
#else
    #define printf
#endif
