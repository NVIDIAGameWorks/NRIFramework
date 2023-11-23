/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#if _WIN32
    #include <io.h>
#endif

#include <array>
#include <map>
#include <functional>

#include <algorithm>
#include <fstream>
#include <filesystem>

#include "Detex/detex.h"

#include "MathLib/MathLib.h"

#include "NRI.h"
#include "Extensions/NRIHelper.h"

#include "Helper.h"
#include "Utils.h"

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

#include "Detex/stb_image.h"

constexpr std::array<const char*, 13> gShaderExts =
{
    "",
    ".vs.",
    ".tcs.",
    ".tes.",
    ".gs.",
    ".fs.",
    ".cs.",
    ".rgen.",
    ".rmiss.",
    "<noimpl>",
    ".rchit.",
    ".rahit.",
    "<noimpl>"
};

//========================================================================================================================
// MISC
//========================================================================================================================
static void GenerateMorphTargetVertices(utils::Scene& scene, const utils::Mesh& mesh, uint32_t morphTargetIndex,
    const uint8_t* positionSrc, size_t positionStride, const uint8_t* normalSrc, size_t normalStride)
{
    std::vector<float3> tangents(mesh.vertexNum, float3::Zero());
    std::vector<float3> bitangents(mesh.vertexNum, float3::Zero());

    for (size_t j = 0; j < mesh.indexNum; j += 3)
    {
        size_t primitiveBaseIndex = mesh.indexOffset + j;

        size_t i0 = scene.indices[primitiveBaseIndex];
        size_t i1 = scene.indices[primitiveBaseIndex + 1];
        size_t i2 = scene.indices[primitiveBaseIndex + 2];

        const utils::UnpackedVertex& v0 = scene.unpackedVertices[mesh.vertexOffset + i0];
        const utils::UnpackedVertex& v1 = scene.unpackedVertices[mesh.vertexOffset + i1];
        const utils::UnpackedVertex& v2 = scene.unpackedVertices[mesh.vertexOffset + i2];

        // base verts
        float3 pb0 = v0.position;
        float3 pb1 = v1.position;
        float3 pb2 = v2.position;

        // src morph target data is delta
        float3 p0 = float3((float *)(positionSrc + positionStride * i0)) + pb0;
        float3 p1 = float3((float *)(positionSrc + positionStride * i1)) + pb1;
        float3 p2 = float3((float *)(positionSrc + positionStride * i2)) + pb2;

        float3 uvEdge20 = float3(v2.uv[0], v2.uv[1], 0.0f) - float3(v0.uv[0], v0.uv[1], 0.0f);
        float3 uvEdge10 = float3(v1.uv[0], v1.uv[1], 0.0f) - float3(v0.uv[0], v0.uv[1], 0.0f);
        
        // base normals
        float3 nb0 = v0.normal;
        float3 nb1 = v1.normal;
        float3 nb2 = v2.normal;
        
        // src morph target data is delta
        float3 n0 = float3((float *)(normalSrc + normalStride * i0)) + nb0;
        float3 n1 = float3((float *)(normalSrc + normalStride * i1)) + nb1;
        float3 n2 = float3((float *)(normalSrc + normalStride * i2)) + nb2;

        // Tangent
        float r = uvEdge10.x * uvEdge20.y - uvEdge20.x * uvEdge10.y;

        float3 tangent, bitangent;
        if (Abs(r) < 1e-9f)
        {
            n1.z += 1e-6f;

            tangent = GetPerpendicularVector(n1);
            bitangent = Cross(n1, tangent);
        }
        else
        {
            float invr = 1.0f / r;

            float3 a = (p1 - p0) * invr;
            float3 b = (p2 - p0) * invr;

            tangent = a * uvEdge20.y - b * uvEdge10.y;
            bitangent = b * uvEdge10.x - a * uvEdge20.x;
        }

        tangents[i0] += tangent;
        tangents[i1] += tangent;
        tangents[i2] += tangent;

        bitangents[i0] += bitangent;
        bitangents[i1] += bitangent;
        bitangents[i2] += bitangent;
    }

    uint32_t vertexOffset = mesh.morphTargetVertexOffset + morphTargetIndex * mesh.vertexNum;
    for (size_t j = 0; j < mesh.vertexNum; j++)
    {
        const utils::UnpackedVertex& v = scene.unpackedVertices[mesh.vertexOffset + j];
        float3 pb = v.position;
        float3 nb = v.normal;

        float3 P = float3((float*)(positionSrc + positionStride * j)) + pb;
        float3 N = float3((float*)(normalSrc + normalStride * j)) + nb;
        float3 T = tangents[j];
        if (Length(T) < 1e-9f)
            T = Cross(bitangents[j], N);
        else // Gram-Schmidt orthogonalize
            T -= N * Dot33(N, T);
        T = Normalize(T);

        // Calculate handedness
        float handedness = Sign(Dot33(Cross(N, T), bitangents[j]));

        // Output
        float2 n = Packed::EncodeUnitVector( N, true );
        float2 t = Packed::EncodeUnitVector( T, true );

        utils::MorphVertex& morphVertex = scene.morphVertices[vertexOffset + j];
        morphVertex.position[0] = Packed::sf2_to_h2(P.x, P.y);
        morphVertex.position[1] = Packed::sf2_to_h2(P.z, handedness);
        morphVertex.normal = Packed::sf2_to_h2(n.x, n.y);
        morphVertex.tangent = Packed::sf2_to_h2(t.x, t.y);
    }
}

static void GeneratePrimitiveDataAndTangents(utils::Scene& scene, const utils::Mesh& mesh)
{
    std::vector<float3> tangents(mesh.vertexNum, float3::Zero());
    std::vector<float3> bitangents(mesh.vertexNum, float3::Zero());

    std::vector<double> curvatures(mesh.vertexNum, 0.0);
    std::vector<double> curvatureWeights(mesh.vertexNum, 0.0);

    for (size_t j = 0; j < mesh.indexNum; j += 3)
    {
        size_t primitiveBaseIndex = mesh.indexOffset + j;

        size_t i0 = scene.indices[primitiveBaseIndex];
        size_t i1 = scene.indices[primitiveBaseIndex + 1];
        size_t i2 = scene.indices[primitiveBaseIndex + 2];

        const utils::UnpackedVertex& v0 = scene.unpackedVertices[mesh.vertexOffset + i0];
        const utils::UnpackedVertex& v1 = scene.unpackedVertices[mesh.vertexOffset + i1];
        const utils::UnpackedVertex& v2 = scene.unpackedVertices[mesh.vertexOffset + i2];

        float3 p0(v0.position);
        float3 p1(v1.position);
        float3 p2(v2.position);

        float3 edge20 = p2 - p0;
        float3 edge10 = p1 - p0;
        float worldArea = Max( Length( Cross(edge20, edge10) ), 1e-9f );

        float3 uvEdge20 = float3(v2.uv[0], v2.uv[1], 0.0f) - float3(v0.uv[0], v0.uv[1], 0.0f);
        float3 uvEdge10 = float3(v1.uv[0], v1.uv[1], 0.0f) - float3(v0.uv[0], v0.uv[1], 0.0f);
        float uvArea = Length( Cross(uvEdge20, uvEdge10) );

        utils::Primitive& primitive = scene.primitives[primitiveBaseIndex / 3];
        primitive.worldToUvUnits = uvArea == 0 ? 1.0f : Sqrt( uvArea / worldArea );

        // Unsigned curvature // TODO: make signed?
        // https://computergraphics.stackexchange.com/questions/1718/what-is-the-simplest-way-to-compute-principal-curvature-for-a-mesh-triangle
        float3 n0 = float3(v0.normal);
        float3 n1 = float3(v1.normal);
        float3 n2 = float3(v2.normal);

        double curvature10 = Abs( Dot33(n1 - n0, p1 - p0) ) / LengthSquared(p1 - p0);
        double curvature21 = Abs( Dot33(n2 - n1, p2 - p1) ) / LengthSquared(p2 - p1);
        double curvature02 = Abs( Dot33(n0 - n2, p0 - p2) ) / LengthSquared(p0 - p2);

        curvatures[i0] += Max(curvature10, curvature02) * worldArea;
        curvatures[i1] += Max(curvature10, curvature21) * worldArea;
        curvatures[i2] += Max(curvature02, curvature21) * worldArea;

        curvatureWeights[i0] += worldArea;
        curvatureWeights[i1] += worldArea;
        curvatureWeights[i2] += worldArea;

        // Tangent
        float r = uvEdge10.x * uvEdge20.y - uvEdge20.x * uvEdge10.y;

        float3 tangent, bitangent;
        if (Abs(r) < 1e-9f)
        {
            n1.z += 1e-6f;

            tangent = GetPerpendicularVector(n1);
            bitangent = Cross(n1, tangent);
        }
        else
        {
            float invr = 1.0f / r;

            float3 a = (p1 - p0) * invr;
            float3 b = (p2 - p0) * invr;

            tangent = a * uvEdge20.y - b * uvEdge10.y;
            bitangent = b * uvEdge10.x - a * uvEdge20.x;
        }

        tangents[i0] += tangent;
        tangents[i1] += tangent;
        tangents[i2] += tangent;

        bitangents[i0] += bitangent;
        bitangents[i1] += bitangent;
        bitangents[i2] += bitangent;
    }

    for (size_t j = 0; j < mesh.vertexNum; j++)
    {
        utils::UnpackedVertex& unpackedVertex = scene.unpackedVertices[mesh.vertexOffset + j];
        float3 N = float3(unpackedVertex.normal);

        float3 T = tangents[j];
        if (Length(T) < 1e-9f)
            T = Cross(bitangents[j], N);
        else // Gram-Schmidt orthogonalize
            T -= N * Dot33(N, T);
        T = Normalize(T);

        // Calculate handedness
        float handedness = Sign( Dot33(Cross(N, T), bitangents[j]) );

        // Output
        float4 result = float4(T.x, T.y, T.z, handedness);
        unpackedVertex.tangent[0] = result.x;
        unpackedVertex.tangent[1] = result.y;
        unpackedVertex.tangent[2] = result.z;
        unpackedVertex.tangent[3] = result.w;
        unpackedVertex.curvature =  float(curvatures[j] / curvatureWeights[j]);

        utils::Vertex& vertex = scene.vertices[mesh.vertexOffset + j];
        vertex.tangent = Packed::uf4_to_uint<10, 10, 10, 2>(result * 0.5f + 0.5f);
    }
}

inline const char* GetShaderExt(nri::GraphicsAPI graphicsAPI)
{
    if (graphicsAPI == nri::GraphicsAPI::D3D11)
        return ".dxbc";
    else if (graphicsAPI == nri::GraphicsAPI::D3D12)
        return ".dxil";

    return ".spirv";
}

static struct FormatMapping
{
    uint32_t detexFormat;
    nri::Format nriFormat;
} formatTable[] = {
    // Uncompressed formats.
    { DETEX_PIXEL_FORMAT_RGB8, nri::Format::UNKNOWN },
    { DETEX_PIXEL_FORMAT_RGBA8, nri::Format::RGBA8_UNORM },
    { DETEX_PIXEL_FORMAT_R8, nri::Format::R8_UNORM },
    { DETEX_PIXEL_FORMAT_SIGNED_R8, nri::Format::R8_SNORM },
    { DETEX_PIXEL_FORMAT_RG8, nri::Format::RG8_UNORM },
    { DETEX_PIXEL_FORMAT_SIGNED_RG8, nri::Format::RG8_SNORM },
    { DETEX_PIXEL_FORMAT_R16, nri::Format::R16_UNORM },
    { DETEX_PIXEL_FORMAT_SIGNED_R16, nri::Format::R16_SNORM },
    { DETEX_PIXEL_FORMAT_RG16, nri::Format::RG16_UNORM },
    { DETEX_PIXEL_FORMAT_SIGNED_RG16, nri::Format::RG16_SNORM },
    { DETEX_PIXEL_FORMAT_RGB16, nri::Format::UNKNOWN },
    { DETEX_PIXEL_FORMAT_RGBA16, nri::Format::RGBA16_UNORM },
    { DETEX_PIXEL_FORMAT_FLOAT_R16, nri::Format::R16_SFLOAT },
    { DETEX_PIXEL_FORMAT_FLOAT_RG16, nri::Format::RG16_SFLOAT },
    { DETEX_PIXEL_FORMAT_FLOAT_RGB16, nri::Format::UNKNOWN },
    { DETEX_PIXEL_FORMAT_FLOAT_RGBA16, nri::Format::RGBA16_SFLOAT },
    { DETEX_PIXEL_FORMAT_FLOAT_R32, nri::Format::R32_SFLOAT },
    { DETEX_PIXEL_FORMAT_FLOAT_RG32, nri::Format::RG32_SFLOAT },
    { DETEX_PIXEL_FORMAT_FLOAT_RGB32, nri::Format::RGB32_SFLOAT },
    { DETEX_PIXEL_FORMAT_FLOAT_RGBA32, nri::Format::RGBA32_SFLOAT },
    { DETEX_PIXEL_FORMAT_A8, nri::Format::UNKNOWN },
    // Compressed formats.
    { DETEX_TEXTURE_FORMAT_BC1, nri::Format::BC1_RGBA_UNORM },
    { DETEX_TEXTURE_FORMAT_BC1A, nri::Format::UNKNOWN },
    { DETEX_TEXTURE_FORMAT_BC2, nri::Format::BC2_RGBA_UNORM },
    { DETEX_TEXTURE_FORMAT_BC3, nri::Format::BC3_RGBA_UNORM },
    { DETEX_TEXTURE_FORMAT_RGTC1, nri::Format::BC4_R_UNORM },
    { DETEX_TEXTURE_FORMAT_SIGNED_RGTC1, nri::Format::BC4_R_SNORM },
    { DETEX_TEXTURE_FORMAT_RGTC2, nri::Format::BC5_RG_UNORM },
    { DETEX_TEXTURE_FORMAT_SIGNED_RGTC2, nri::Format::BC5_RG_SNORM },
    { DETEX_TEXTURE_FORMAT_BPTC_FLOAT, nri::Format::BC6H_RGB_UFLOAT },
    { DETEX_TEXTURE_FORMAT_BPTC_SIGNED_FLOAT, nri::Format::BC6H_RGB_SFLOAT },
    { DETEX_TEXTURE_FORMAT_BPTC, nri::Format::BC7_RGBA_UNORM },
    { DETEX_TEXTURE_FORMAT_ETC1, nri::Format::UNKNOWN },
    { DETEX_TEXTURE_FORMAT_ETC2, nri::Format::UNKNOWN },
    { DETEX_TEXTURE_FORMAT_ETC2_PUNCHTHROUGH, nri::Format::UNKNOWN },
    { DETEX_TEXTURE_FORMAT_ETC2_EAC, nri::Format::UNKNOWN },
    { DETEX_TEXTURE_FORMAT_EAC_R11, nri::Format::UNKNOWN },
    { DETEX_TEXTURE_FORMAT_EAC_SIGNED_R11, nri::Format::UNKNOWN },
    { DETEX_TEXTURE_FORMAT_EAC_RG11, nri::Format::UNKNOWN },
    { DETEX_TEXTURE_FORMAT_EAC_SIGNED_RG11, nri::Format::UNKNOWN }
};

static nri::Format GetFormatNRI(uint32_t detexFormat)
{
    for (auto& entry : formatTable)
    {
        if (entry.detexFormat == detexFormat)
            return entry.nriFormat;
    }

    return nri::Format::UNKNOWN;
}

static nri::Format MakeSRGBFormat(nri::Format format)
{
    switch (format)
    {
    case nri::Format::RGBA8_UNORM:
        return nri::Format::RGBA8_SRGB;

    case nri::Format::BC1_RGBA_UNORM:
        return nri::Format::BC1_RGBA_SRGB;

    case nri::Format::BC2_RGBA_UNORM:
        return nri::Format::BC2_RGBA_SRGB;

    case nri::Format::BC3_RGBA_UNORM:
        return nri::Format::BC3_RGBA_SRGB;

    case nri::Format::BC7_RGBA_UNORM:
        return nri::Format::BC7_RGBA_SRGB;

    default:
        return format;
    }
}

//========================================================================================================================
// TEXTURE
//========================================================================================================================

inline detexTexture** ToTexture(utils::Mip* mips)
{
    return (detexTexture**)mips;
}

inline detexTexture* ToMip(utils::Mip mip)
{
    return (detexTexture*)mip;
}

utils::Texture::~Texture()
{
    detexFreeTexture(ToTexture(mips), mipNum);
}

void utils::Texture::GetSubresource(nri::TextureSubresourceUploadDesc& subresource, uint32_t mipIndex, uint32_t arrayIndex) const
{
    // TODO: 3D images are not supported, "subresource.slices" needs to be allocated to store pointers to all slices of the current mipmap
    assert(GetDepth() == 1);
    PLATFORM_UNUSED(arrayIndex);

    detexTexture* mip = ToMip(mips[mipIndex]);

    int rowPitch, slicePitch;
    detexComputePitch(mip->format, mip->width, mip->height, &rowPitch, &slicePitch);

    subresource.slices = mip->data;
    subresource.sliceNum = 1;
    subresource.rowPitch = (uint32_t)rowPitch;
    subresource.slicePitch = (uint32_t)slicePitch;
}

bool utils::Texture::IsBlockCompressed() const
{
    return detexFormatIsCompressed( ToMip(mips[0])->format );
}

const char* utils::GetFileName(const std::string& path)
{
    const size_t slashPos = path.find_last_of("\\/");
    if (slashPos != std::string::npos)
        return path.c_str() + slashPos + 1;

    return "";
}

//========================================================================================================================
// UTILS
//========================================================================================================================

std::string utils::GetFullPath(const std::string& localPath, DataFolder dataFolder)
{
    std::string path = "_Data/"; // it's a symbolic link
    if (dataFolder == DataFolder::SHADERS)
        path = "_Shaders/"; // special folder with generated files
    else if (dataFolder == DataFolder::TEXTURES)
        path += "Textures/";
    else if (dataFolder == DataFolder::SCENES)
        path += "Scenes/";
    else if (dataFolder == DataFolder::TESTS)
        path = "Tests/"; // special folder stored in Git

    return path + localPath;
}

bool utils::LoadFile(const std::string& path, std::vector<uint8_t>& data)
{
    FILE* file = fopen(path.c_str(), "rb");

    if (file == nullptr)
    {
        printf("ERROR: File '%s' is not found!\n", path.c_str());
        data.clear();
        return false;
    }

    printf("Loading file '%s'...\n", GetFileName(path));

    fseek(file, 0, SEEK_END);
    const size_t size = ftell(file); // 32-bit size
    fseek(file, 0, SEEK_SET);

    data.resize(size);
    const size_t readSize = fread(&data[0], size, 1, file);
    fclose(file);

    return !data.empty() && readSize == 1;
}

nri::ShaderDesc utils::LoadShader(nri::GraphicsAPI graphicsAPI, const std::string& shaderName, ShaderCodeStorage& storage, const char* entryPointName)
{
    const char* ext = GetShaderExt(graphicsAPI);
    std::string path = GetFullPath(shaderName + ext, DataFolder::SHADERS);
    nri::ShaderDesc shaderDesc = {};

    uint32_t i = 1;
    for (; i < (uint32_t)gShaderExts.size(); i++)
    {
        if (path.rfind(gShaderExts[i]) != std::string::npos)
        {
            storage.push_back( std::vector<uint8_t>() );
            std::vector<uint8_t>& code = storage.back();

            if (LoadFile(path, code))
            {
                shaderDesc.stage = (nri::ShaderStage)i;
                shaderDesc.bytecode = code.data();
                shaderDesc.size = code.size();
                shaderDesc.entryPointName = entryPointName;
            }

            break;
        }
    }

    if (i == (uint32_t)nri::ShaderStage::MAX_NUM)
    {
        printf("ERROR: Shader '%s' has invalid shader extension!\n", shaderName.c_str());

        NRI_ABORT_ON_FALSE(false);
    };

    return shaderDesc;
}

namespace utils
{
static void PostProcessTexture(const std::string &name, Texture& texture, bool computeAvgColorAndAlphaMode, detexTexture** dTexture, int mipNum)
{
    texture.mips = (Mip*)dTexture;
    texture.name = name;
    texture.format = GetFormatNRI(dTexture[0]->format);
    texture.width = (uint16_t)dTexture[0]->width;
    texture.height = (uint16_t)dTexture[0]->height;
    texture.mipNum = (uint16_t)mipNum;

    // TODO: detex doesn't support cubemaps and 3D textures
    texture.arraySize = 1;
    texture.depth = 1;

    texture.alphaMode = AlphaMode::OPAQUE;
    if (computeAvgColorAndAlphaMode)
    {
        // Alpha mode
        if (texture.format == nri::Format::BC1_RGBA_UNORM || texture.format == nri::Format::BC1_RGBA_SRGB)
        {
            bool hasTransparency = false;
            for (int i = mipNum - 1; i >= 0 && !hasTransparency; i--) {
                const size_t size = detexTextureSize(dTexture[i]->width_in_blocks, dTexture[i]->height_in_blocks, dTexture[i]->format);
                const uint8_t* bc1 = dTexture[i]->data;

                for (size_t j = 0; j < size && !hasTransparency; j += 8)
                {
                    const uint16_t* c = (uint16_t*)bc1;
                    if (c[0] <= c[1])
                    {
                        const uint32_t bits = *(uint32_t*)(bc1 + 4);
                        for (uint32_t k = 0; k < 32 && !hasTransparency; k += 2)
                            hasTransparency = ((bits >> k) & 0x3) == 0x3;
                    }
                    bc1 += 8;
                }
            }

            if (hasTransparency)
                texture.alphaMode = AlphaMode::PREMULTIPLIED;
        }

        // Decompress last mip
        std::vector<uint8_t> image;
        detexTexture* lastMip = dTexture[mipNum - 1];
        uint8_t* rgba8 = lastMip->data;
        if (lastMip->format != DETEX_PIXEL_FORMAT_RGBA8)
        {
            // Convert to RGBA8 if the texture is compressed
            image.resize(lastMip->width * lastMip->height * detexGetPixelSize(DETEX_PIXEL_FORMAT_RGBA8));
            detexDecompressTextureLinear(lastMip, &image[0], DETEX_PIXEL_FORMAT_RGBA8);
            rgba8 = &image[0];
        }

        // Average color
        float4 avgColor = float4::Zero();
        const size_t pixelNum = lastMip->width * lastMip->height;
        for (size_t i = 0; i < pixelNum; i++)
            avgColor += Packed::uint_to_uf4<8, 8, 8, 8>(*(uint32_t*)(rgba8 + i * 4));
        avgColor /= float(pixelNum);

        if (texture.alphaMode != AlphaMode::PREMULTIPLIED && avgColor.w < 254.0f / 255.0f)
            texture.alphaMode = AlphaMode::TRANSPARENT;

        if (texture.alphaMode == AlphaMode::TRANSPARENT && avgColor.w == 0.0f)
        {
            printf("WARNING: Texture '%s' is fully transparent!\n", name.c_str());
            texture.alphaMode = AlphaMode::OFF;
        }
    }
}
}

bool utils::LoadTextureFromMemory(const std::string& name, const uint8_t* data, int dataSize, 
    Texture& texture, bool computeAvgColorAndAlphaMode)
{
    printf("Loading embedded texture '%s'...\n", name.c_str());

    int x, y, comp;
    unsigned char* image = stbi_load_from_memory((stbi_uc const*)data, dataSize, &x, &y, &comp, STBI_rgb_alpha);
    if (!image)
    {
        printf("Could not read memory for embedded texture %s. Reason: %s", name.c_str(), stbi_failure_reason());
        return false;
    }
    detexTexture** dTexture = (detexTexture**)malloc(sizeof(detexTexture*));
    dTexture[0] = (detexTexture*)malloc(sizeof(detexTexture));
    dTexture[0]->format = DETEX_PIXEL_FORMAT_RGBA8;
    dTexture[0]->width = x;
    dTexture[0]->height = y;
    dTexture[0]->width_in_blocks = x;
    dTexture[0]->height_in_blocks = y;
    size_t size = x * y * detexGetPixelSize(DETEX_PIXEL_FORMAT_RGBA8);
    dTexture[0]->data = (uint8_t*)malloc(size);
    memcpy(dTexture[0]->data, image, size);
    stbi_image_free(image);
    
    const int kMipNum = 1;
    PostProcessTexture(name, texture, computeAvgColorAndAlphaMode, dTexture, kMipNum);
    return true;
}

bool utils::LoadTexture(const std::string& path, Texture& texture, bool computeAvgColorAndAlphaMode)
{
    printf("Loading texture '%s'...\n", GetFileName(path));

    detexTexture** dTexture = nullptr;
    int mipNum = 0;

    if (!detexLoadTextureFileWithMipmaps(path.c_str(), 32, &dTexture, &mipNum))
    {
        printf("ERROR: Can't load texture '%s'\n", path.c_str());

        return false;
    }

    PostProcessTexture(path, texture, computeAvgColorAndAlphaMode, dTexture, mipNum);
    
    return true;
}

void utils::LoadTextureFromMemory(nri::Format format, uint32_t width, uint32_t height, const uint8_t *pixels, Texture &texture)
{
    assert(format == nri::Format::R8_UNORM);

    detexTexture **dTexture;
    detexLoadTextureFromMemory(DETEX_PIXEL_FORMAT_R8, width, height, pixels, &dTexture);

    texture.mipNum = 1;
    texture.arraySize = 1;
    texture.depth = 1;
    texture.format = format;
    texture.alphaMode = AlphaMode::OPAQUE;
    texture.mips = (Mip*)dTexture;
}

static const char* cgltfErrorToString(cgltf_result res)
{
    switch (res)
    {
        case cgltf_result_success:
            return "Success";
        case cgltf_result_data_too_short:
            return "Data is too short";
        case cgltf_result_unknown_format:
            return "Unknown format";
        case cgltf_result_invalid_json:
            return "Invalid JSON";
        case cgltf_result_invalid_gltf:
            return "Invalid GLTF";
        case cgltf_result_invalid_options:
            return "Invalid options";
        case cgltf_result_file_not_found:
            return "File not found";
        case cgltf_result_io_error:
            return "I/O error";
        case cgltf_result_out_of_memory:
            return "Out of memory";
        case cgltf_result_legacy_gltf:
            return "Legacy GLTF";
        default:
            return "Unknown error";
    }
}

static std::pair<const uint8_t*, size_t> cgltfBufferIterator(const cgltf_accessor* accessor, size_t defaultStride)
{
    // TODO: sparse accessor support
    const cgltf_buffer_view* view = accessor->buffer_view;
    const uint8_t* data = (uint8_t*)view->buffer->data + view->offset + accessor->offset;
    const size_t stride = view->stride ? view->stride : defaultStride;

    return std::make_pair(data, stride);
}

// GLTF only support DDS images through the MSFT_texture_dds extension.
// Since cgltf does not support this extension, we parse the custom extension string as json here.
// See https://github.com/KhronosGroup/GLTF/tree/master/extensions/2.0/Vendor/MSFT_texture_dds
static const cgltf_image* ParseDdsImage(const cgltf_texture* texture, const cgltf_data* objects)
{
    for (size_t i = 0; i < texture->extensions_count; i++)
    {
        const cgltf_extension& ext = texture->extensions[i];

        if (!ext.name || !ext.data)
            continue;

        if (strcmp(ext.name, "MSFT_texture_dds") != 0)
            continue;

        size_t extensionLength = strlen(ext.data);
        if (extensionLength > 1024)
            return nullptr; // safeguard against weird inputs

        jsmn_parser parser;
        jsmn_init(&parser);

        // count the tokens, normally there are 3
        int numTokens = jsmn_parse(&parser, ext.data, extensionLength, nullptr, 0);

        // allocate the tokens on the stack
        jsmntok_t* tokens = (jsmntok_t*)alloca(numTokens * sizeof(jsmntok_t));

        // reset the parser and prse
        jsmn_init(&parser);
        int numParsed = jsmn_parse(&parser, ext.data, extensionLength, tokens, numTokens);
        if (numParsed != numTokens)
            goto fail;

        if (tokens[0].type != JSMN_OBJECT)
            goto fail; // expecting that the extension is an object

        for (int k = 1; k < numTokens; k++)
        {
            if (tokens[k].type != JSMN_STRING)
                goto fail; // expecting a string key

            if (cgltf_json_strcmp(tokens + k, (const uint8_t*)ext.data, "source") == 0)
            {
                ++k;
                int index = cgltf_json_to_int(tokens + k, (const uint8_t*)ext.data);
                if (index < 0)
                    goto fail; // expecting a non-negative integer; non-value results in CGLTF_ERROR_JSON which is negative

                if (size_t(index) >= objects->images_count)
                {
                    printf("WARNING: Invalid image index %d specified in GLTF texture definition\n", index);
                    return nullptr;
                }

                return objects->images + index;
            }

            // this was something else - skip it
            k = cgltf_skip_json(tokens, k);
        }

    fail:
        printf("WARNING: Failed to parse the DDS GLTF extension: %s\n", ext.data);
        return nullptr;
    }

    return nullptr;
}

void DecomposeAffine(const float4x4& transform, float3& translation, float4& rotation, float3& scale)
{
    translation = transform.col3;

    float3 col0 = transform.col0;
    float3 col1 = transform.col1;
    float3 col2 = transform.col2;

    scale.x = Length(col0);
    scale.y = Length(col1);
    scale.z = Length(col2);
    if (scale.x > 0.f) col0 /= scale.x;
    if (scale.y > 0.f) col1 /= scale.y;
    if (scale.z > 0.f) col2 /= scale.z;

    float3 zAxis = Cross(col0, col1);
    if (Dot33(zAxis, col2) < 0.0f)
    {
        scale.x = -scale.x;
        col0 = -col0;
    }

    // https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    rotation.w = Sqrt(Max(0.0f, 1.0f + col0.x + col1.y + col2.z)) * 0.5f;
    rotation.x = Sqrt(Max(0.0f, 1.0f + col0.x - col1.y - col2.z)) * 0.5f;
    rotation.y = Sqrt(Max(0.0f, 1.0f - col0.x + col1.y - col2.z)) * 0.5f;
    rotation.z = Sqrt(Max(0.0f, 1.0f - col0.x - col1.y + col2.z)) * 0.5f;
    rotation.x = std::copysign(rotation.x, col2.y - col1.z);
    rotation.y = std::copysign(rotation.y, col0.z - col2.x);
    rotation.z = std::copysign(rotation.z, col1.x - col0.y);
}

bool utils::LoadScene(const std::string& path, Scene& scene, bool allowUpdate)
{
    printf("Loading scene '%s'...\n", GetFileName(path));

    std::filesystem::path normPath(path.c_str());
    normPath = std::filesystem::canonical(normPath);

    cgltf_options options{};

    cgltf_data* objects{};
    cgltf_result res = cgltf_parse_file(&options, path.c_str(), &objects);
    if (res != cgltf_result_success)
    {
        printf("Couldn't load GLTF file '%s': %s", path.c_str(), cgltfErrorToString(res));
        return false;
    }

    res = cgltf_load_buffers(&options, objects, path.c_str());
    if (res != cgltf_result_success)
    {
        printf("Failed to load buffers for GLTF file '%s': %s", path.c_str(), cgltfErrorToString(res));
        return false;
    }

    // Meshes
    // TODO: framework doesn't support multiple submeshes per instance, treat every primitive as a separate mesh
    std::vector<std::vector<size_t>> meshesPrimMap;
    meshesPrimMap.resize(objects->meshes_count);

    size_t meshNum = 0;
    for (size_t mesh_idx = 0; mesh_idx < objects->meshes_count; mesh_idx++)
    {
        const cgltf_mesh& gltfMesh = objects->meshes[mesh_idx];

        std::vector<size_t>& meshPrimMap = meshesPrimMap[mesh_idx];
        meshPrimMap.resize(gltfMesh.primitives_count);

        for (size_t prim_idx = 0; prim_idx < gltfMesh.primitives_count; prim_idx++)
        {
            const cgltf_primitive& gltfSubmesh = gltfMesh.primitives[prim_idx];
            if (gltfSubmesh.type != cgltf_primitive_type_triangles || gltfSubmesh.attributes_count == 0)
                continue;

            meshPrimMap[prim_idx] = meshNum++;
        }
    }

    size_t materialNum = objects->materials_count;
    size_t materialOffset = scene.materials.size();
    scene.materials.resize(materialOffset + materialNum);

    uint32_t meshOffset = (uint32_t)scene.meshes.size();
    scene.meshes.resize(meshOffset + meshNum);

    size_t totalIndices = scene.indices.size();
    size_t totalVertices = scene.vertices.size();
    size_t totalMorphMeshVertices = scene.morphVertices.size();

    for (size_t mesh_idx = 0; mesh_idx < objects->meshes_count; mesh_idx++)
    {
        const cgltf_mesh& gltfMesh = objects->meshes[mesh_idx];

        for (size_t prim_idx = 0; prim_idx < gltfMesh.primitives_count; prim_idx++)
        {
            const cgltf_primitive& gltfSubmesh = gltfMesh.primitives[prim_idx];
            if (gltfSubmesh.type != cgltf_primitive_type_triangles || gltfSubmesh.attributes_count == 0)
                continue;

            size_t meshVertices = 0;
            
            // search for position, first attribute may not be position and may have different size if malformed
            for (size_t attr_idx = 0; attr_idx < gltfSubmesh.attributes_count; attr_idx++)
            {
                const cgltf_attribute& attr = gltfSubmesh.attributes[attr_idx];
                if (attr.type == cgltf_attribute_type_position)
                {
                    meshVertices = attr.data->count;
                    break;
                }
            }
                    
            if (meshVertices == 0)
                continue;

            size_t meshIndices = gltfSubmesh.indices ? gltfSubmesh.indices->count : meshVertices;
            uint32_t meshIndex = meshOffset + (uint32_t)meshesPrimMap[mesh_idx][prim_idx];

            Mesh& mesh = scene.meshes[meshIndex];
            mesh.indexOffset = (uint32_t)totalIndices;
            mesh.vertexOffset = (uint32_t)totalVertices;
            mesh.indexNum = (uint32_t)meshIndices;
            mesh.vertexNum = (uint32_t)meshVertices;

            totalIndices += mesh.indexNum;
            totalVertices += mesh.vertexNum;

            bool hasMorphTargets = gltfSubmesh.targets_count > 0;
            for (uint32_t target_idx = 0; target_idx < gltfSubmesh.targets_count; target_idx++)
            {
                const cgltf_morph_target &morphTarget = gltfSubmesh.targets[target_idx];
                if (morphTarget.attributes_count == 0)
                {
                    hasMorphTargets = false;
                    break;
                }

                bool hasPositions = false;
                bool hasNormals = false;
                for (uint32_t attr_idx = 0; attr_idx < morphTarget.attributes_count; attr_idx++)
                {
                    if (morphTarget.attributes[attr_idx].type == cgltf_attribute_type_position)
                    {
                        hasPositions = morphTarget.attributes[attr_idx].data->count == mesh.vertexNum;
                    }
                    else if (morphTarget.attributes[attr_idx].type == cgltf_attribute_type_normal)
                    {
                        hasNormals = morphTarget.attributes[attr_idx].data->count == mesh.vertexNum;
                    }
                }

                if (!hasPositions || !hasNormals)
                {
                    hasMorphTargets = false;
                    break;
                }
            }

            if (hasMorphTargets)
            {
                mesh.morphMeshIndexOffset = scene.morphMeshTotalIndicesNum;
                mesh.morphTargetVertexOffset = (uint32_t)totalMorphMeshVertices;
                mesh.morphTargetNum = (uint32_t)gltfSubmesh.targets_count;

                scene.morphMeshTotalIndicesNum += mesh.indexNum;
                totalMorphMeshVertices += mesh.vertexNum * mesh.morphTargetNum;

                scene.morphMeshes.push_back(meshIndex);
            }
        }
    }

    scene.indices.resize(totalIndices);
    scene.primitives.resize(totalIndices / 3);
    scene.vertices.resize(totalVertices);
    scene.unpackedVertices.resize(totalVertices);
    scene.morphVertices.resize(totalMorphMeshVertices);

    // Geometry
    for (size_t mesh_idx = 0; mesh_idx < objects->meshes_count; mesh_idx++)
    {
        const cgltf_mesh& gltfMesh = objects->meshes[mesh_idx];

        for (size_t prim_idx = 0; prim_idx < gltfMesh.primitives_count; prim_idx++)
        {
            const cgltf_primitive& gltfSubmesh = gltfMesh.primitives[prim_idx];
            if (gltfSubmesh.type != cgltf_primitive_type_triangles || gltfSubmesh.attributes_count == 0)
                continue;

            size_t meshIndex = meshOffset + meshesPrimMap[mesh_idx][prim_idx];
            Mesh& mesh = scene.meshes[meshIndex];
            mesh.aabb.Clear();

            const cgltf_accessor* positions = nullptr;
            const cgltf_accessor* normals = nullptr;
            const cgltf_accessor* texcoords = nullptr;
            const cgltf_accessor* joint_weights = nullptr;
            const cgltf_accessor* joint_indices = nullptr;

            for (size_t attr_idx = 0; attr_idx < gltfSubmesh.attributes_count; attr_idx++)
            {
                const cgltf_attribute& attr = gltfSubmesh.attributes[attr_idx];

                switch (attr.type)
                {
                    case cgltf_attribute_type_position:
                        assert(attr.data->type == cgltf_type_vec3);
                        assert(attr.data->component_type == cgltf_component_type_r_32f);
                        positions = attr.data;
                        break;
                    case cgltf_attribute_type_normal:
                        assert(attr.data->type == cgltf_type_vec3);
                        assert(attr.data->component_type == cgltf_component_type_r_32f);
                        normals = attr.data;
                        break;
                    case cgltf_attribute_type_texcoord:
                        assert(attr.data->type == cgltf_type_vec2);
                        assert(attr.data->component_type == cgltf_component_type_r_32f);
                        if (attr.index == 0)
                            texcoords = attr.data;
                        break;
                    case cgltf_attribute_type_joints:
                        assert(attr.data->type == cgltf_type_vec4);
                        assert(attr.data->component_type == cgltf_component_type_r_8u || attr.data->component_type == cgltf_component_type_r_16u);
                        joint_indices = attr.data;
                        break;
                    case cgltf_attribute_type_weights:
                        assert(attr.data->type == cgltf_type_vec4);
                        assert(attr.data->component_type == cgltf_component_type_r_8u || attr.data->component_type == cgltf_component_type_r_16u || attr.data->component_type == cgltf_component_type_r_32f);
                        joint_weights = attr.data;
                        break;
                }
            }

            assert(positions);
            assert(mesh.vertexNum == positions->count);

            if (gltfSubmesh.indices)
            { // indexed geometry
                assert(gltfSubmesh.indices->component_type == cgltf_component_type_r_32u || gltfSubmesh.indices->component_type == cgltf_component_type_r_16u || gltfSubmesh.indices->component_type == cgltf_component_type_r_8u);
                assert(gltfSubmesh.indices->type == cgltf_type_scalar);

                auto [indexSrc, indexStride] = cgltfBufferIterator(gltfSubmesh.indices, 0);

                switch (gltfSubmesh.indices->component_type)
                {
                    case cgltf_component_type_r_8u:
                        if (!indexStride)
                            indexStride = sizeof(uint8_t);

                        for (size_t i_idx = 0; i_idx < mesh.indexNum; i_idx++)
                        {
                            scene.indices[mesh.indexOffset + i_idx] = (Index)(*(const uint8_t*)indexSrc);
                            indexSrc += indexStride;
                        }
                        break;
                    case cgltf_component_type_r_16u:
                        if (!indexStride)
                            indexStride = sizeof(uint16_t);

                        for (size_t i_idx = 0; i_idx < mesh.indexNum; i_idx++)
                        {
                            scene.indices[mesh.indexOffset + i_idx] = (Index)(*(const uint16_t*)indexSrc);
                            indexSrc += indexStride;
                        }
                        break;
                    case cgltf_component_type_r_32u:
                        if (!indexStride)
                            indexStride = sizeof(uint32_t);

                        for (size_t i_idx = 0; i_idx < mesh.indexNum; i_idx++)
                        {
                            scene.indices[mesh.indexOffset + i_idx] = (Index)(*(const uint32_t*)indexSrc);
                            indexSrc += indexStride;
                        }
                        break;
                    default:
                        assert(false);
                }
            }
            else
            { // unindexed geometry
                for (size_t i_idx = 0; i_idx < mesh.vertexNum; i_idx++)
                    scene.indices[mesh.indexOffset + i_idx] = (Index)i_idx;
            }

            if (positions)
            {
                auto [positionSrc, positionStride] = cgltfBufferIterator(positions, sizeof(float) * 3);

                for (size_t v_idx = 0; v_idx < mesh.vertexNum; v_idx++)
                {
                    float3 position((const float*)positionSrc);

                    UnpackedVertex& unpackedVertex = scene.unpackedVertices[mesh.vertexOffset + v_idx];
                    unpackedVertex.position[0] = position.x;
                    unpackedVertex.position[1] = position.y;
                    unpackedVertex.position[2] = position.z;

                    Vertex& vertex = scene.vertices[mesh.vertexOffset + v_idx];
                    vertex.position[0] = position.x;
                    vertex.position[1] = position.y;
                    vertex.position[2] = position.z;

                    mesh.aabb.Add(position);

                    positionSrc += positionStride;
                }
            }

            if (normals)
            {
                assert(normals->count == positions->count);

                auto [normalSrc, normalStride] = cgltfBufferIterator(normals, sizeof(float) * 3);

                for (size_t v_idx = 0; v_idx < mesh.vertexNum; v_idx++)
                {
                    float3 normal((const float*)normalSrc);

                    UnpackedVertex& unpackedVertex = scene.unpackedVertices[mesh.vertexOffset + v_idx];
                    unpackedVertex.normal[0] = normal.x;
                    unpackedVertex.normal[1] = normal.y;
                    unpackedVertex.normal[2] = normal.z;

                    Vertex& vertex = scene.vertices[mesh.vertexOffset + v_idx];
                    vertex.normal = Packed::uf4_to_uint<10, 10, 10, 2>(normal * 0.5f + 0.5f);

                    normalSrc += normalStride;
                }
            }

            if (texcoords)
            {
                assert(texcoords->count == positions->count);

                auto [texcoordSrc, texcoordStride] = cgltfBufferIterator(texcoords, sizeof(float) * 2);

                for (size_t v_idx = 0; v_idx < mesh.vertexNum; v_idx++)
                {
                    const float* uv = (const float*)texcoordSrc;

                    float u = Min(uv[0], 65504.0f);
                    float v = Min(uv[1], 65504.0f);

                    UnpackedVertex& unpackedVertex = scene.unpackedVertices[mesh.vertexOffset + v_idx];
                    unpackedVertex.uv[0] = u;
                    unpackedVertex.uv[1] = v;

                    Vertex& vertex = scene.vertices[mesh.vertexOffset + v_idx];
                    vertex.uv = Packed::sf2_to_h2(u, v);

                    texcoordSrc += texcoordStride;
                }
            }
            else
            {
                for (size_t v_idx = 0; v_idx < mesh.vertexNum; v_idx++)
                {
                    UnpackedVertex& unpackedVertex = scene.unpackedVertices[mesh.vertexOffset + v_idx];
                    unpackedVertex.uv[0] = 0.0f;
                    unpackedVertex.uv[1] = 0.0f;

                    Vertex& vertex = scene.vertices[mesh.vertexOffset + v_idx];
                    vertex.uv = 0;
                }
            }

            for (uint32_t target_idx = 0; target_idx < gltfSubmesh.targets_count; target_idx++)
            {
                const cgltf_morph_target& target = gltfSubmesh.targets[target_idx];
                const cgltf_accessor* target_positions = nullptr;
                const cgltf_accessor* target_normals = nullptr;

                for (size_t attr_idx = 0; attr_idx < target.attributes_count; attr_idx++)
                {
                    const cgltf_attribute& attr = target.attributes[attr_idx];

                    switch (attr.type)
                    {
                    case cgltf_attribute_type_position:
                        assert(attr.data->type == cgltf_type_vec3);
                        assert(attr.data->component_type == cgltf_component_type_r_32f);
                        target_positions = attr.data;
                        break;
                    case cgltf_attribute_type_normal:
                        assert(attr.data->type == cgltf_type_vec3);
                        assert(attr.data->component_type == cgltf_component_type_r_32f);
                        target_normals = attr.data;
                        break;
                    }
                }

                assert(target_positions && target_normals);

                auto [positionSrc, positionStride] = cgltfBufferIterator(target_positions, sizeof(float) * 3);
                auto [normalSrc, normalStride] = cgltfBufferIterator(target_normals, sizeof(float) * 3);
                
                GenerateMorphTargetVertices(scene, mesh, target_idx, positionSrc, positionStride, normalSrc, normalStride);
            }

            // Per primitive data and tangents
            GeneratePrimitiveDataAndTangents(scene, mesh);
        }
    }

    // Walk through the nodes and fill instances
    const uint32_t instanceOffset = (uint32_t)scene.instances.size();
    scene.instances.reserve(instanceOffset + objects->nodes_count);

    size_t currentSceneMeshCount = scene.meshes.size() - meshOffset;
    std::vector<uint32_t> sharedMeshInstanceIndices(currentSceneMeshCount, InvalidIndex);

    std::map<cgltf_node*, std::vector<uint32_t>> nodeToInstanceMap;

    auto AddMeshInstance = [&scene, &sharedMeshInstanceIndices, meshOffset](uint32_t meshIndex)
    {
        Mesh& mesh = scene.meshes[meshIndex];

        uint32_t currentSceneMeshIndex = meshIndex - meshOffset;
        uint32_t meshInstanceIndex = (uint32_t)scene.meshInstances.size();
        if (!mesh.HasMorphTargets())
        {
            // check if we already made a sharable mesh instance for this mesh
            if (sharedMeshInstanceIndices[currentSceneMeshIndex] != InvalidIndex)
                return sharedMeshInstanceIndices[currentSceneMeshIndex];

            // Update cache for new mesh Instance Index
            sharedMeshInstanceIndices[currentSceneMeshIndex] = meshInstanceIndex;
        }
        
        scene.meshInstances.push_back({});

        MeshInstance& meshInstance = scene.meshInstances.back();
        meshInstance.meshIndex = meshIndex;
        meshInstance.primitiveOffset = scene.totalInstancedPrimitivesNum;
        meshInstance.blasIndex = InvalidIndex;

        uint32_t numPrimitives = mesh.indexNum / 3;
        if (mesh.HasMorphTargets())
        {
            meshInstance.morphedVertexOffset = scene.morphedVerticesNum;
            scene.morphedVerticesNum += mesh.vertexNum;

            meshInstance.morphedPrimitiveOffset = scene.morphedPrimitivesNum;
            scene.morphedPrimitivesNum += numPrimitives;
        }

        scene.totalInstancedPrimitivesNum += numPrimitives;

        return meshInstanceIndex;
    };

    std::function<void(cgltf_node*, float4x4)> traverseNode = [&](cgltf_node* node, const float4x4& parentTransform)
    {
        float4x4 worldTransform;
        if (node->has_matrix)
        {
            const auto& tr = node->matrix;
            float4x4 transform
            (
                tr[0], tr[4], tr[8], tr[12],
                tr[1], tr[5], tr[9], tr[13],
                tr[2], tr[6], tr[10], tr[14],
                tr[3], tr[7], tr[11], tr[15]
            );
            worldTransform = parentTransform * transform;
        }
        else
        {
            float4x4 scale = float4x4::Identity();
            if (node->has_scale)
                scale.SetupByScale(node->scale);

            float4x4 rotation = float4x4::Identity();
            if (node->has_rotation)
                rotation.SetupByQuaternion(node->rotation);

            float4x4 translation = float4x4::Identity();
            if (node->has_translation)
                translation.SetupByTranslation(node->translation);

            float4x4 localTransform = translation * (rotation * scale);
            worldTransform = parentTransform * localTransform;
        }

        if (node->mesh)
        {
            float4x4 transform = worldTransform;
            double3 position = ToDouble(transform.GetCol3().To3d());
            transform.SetTranslation( float3::Zero() );

            size_t meshIndex = node->mesh - objects->meshes;

            std::vector<uint32_t>& vec = nodeToInstanceMap[node];

            for (uint32_t primIndex = 0; primIndex < node->mesh->primitives_count; ++primIndex)
            {
                const cgltf_primitive& gltfSubmesh = node->mesh->primitives[primIndex];

                size_t materialIndex = gltfSubmesh.material ? (gltfSubmesh.material - objects->materials) : 0;
                size_t remappedMeshIndex = meshOffset + meshesPrimMap[meshIndex][primIndex];

                Instance& instance = scene.instances.emplace_back();
                Mesh& m = scene.meshes[remappedMeshIndex];

                instance.meshInstanceIndex = AddMeshInstance((uint32_t)remappedMeshIndex);
                instance.position = position;
                instance.rotation = transform;
                instance.materialIndex = (uint32_t)(materialOffset + materialIndex);
                instance.allowUpdate = allowUpdate || m.HasMorphTargets();

                vec.push_back((uint32_t)(scene.instances.size() - 1));

                cBoxf aabb;
                TransformAabb(worldTransform, m.aabb, aabb);

                scene.aabb.Add(aabb);
            }
        }

        for (cgltf_size nodeIndex = 0; nodeIndex < node->children_count; ++nodeIndex)
            traverseNode(node->children[nodeIndex], worldTransform);
    };

    // GLTF models expect Y up whereas framework has Z up
    scene.mSceneToWorld.SetupByRotationX( Pi(0.5f) );

    for (cgltf_size nodeIndex = 0; nodeIndex < objects->scene->nodes_count; ++nodeIndex)
        traverseNode(objects->scene->nodes[nodeIndex], scene.mSceneToWorld);

    // TODO: properly update "allowUpdate"
    if (objects->animations_count)
    {
        for (uint32_t animIndex = 0; animIndex < objects->animations_count; ++animIndex)
        {
            cgltf_animation* gltfAnim = objects->animations + animIndex;

            scene.animations.push_back(Animation());
            Animation& animation = scene.animations.back();
            animation.name = gltfAnim->name ? gltfAnim->name : "";

            { // Setup scene graph
                animation.sceneNodes.resize(objects->nodes_count);
                for (uint32_t nodeIndex = 0; nodeIndex < objects->nodes_count; ++nodeIndex)
                {
                    cgltf_node* gltfNode = objects->nodes + nodeIndex;
                    SceneNode& sceneNode = animation.sceneNodes[nodeIndex];

                    sceneNode.children.resize(gltfNode->children_count);
                    for (uint32_t childIndex = 0; childIndex < gltfNode->children_count; ++childIndex)
                    {
                        uint32_t index = (uint32_t)(gltfNode->children[childIndex] - objects->nodes);
                        sceneNode.children[childIndex] = &animation.sceneNodes[index];
                    }

                    if (gltfNode->has_matrix)
                    {
                        const auto& tr = gltfNode->matrix;
                        float4x4 transform
                        (
                            tr[0], tr[4], tr[8], tr[12],
                            tr[1], tr[5], tr[9], tr[13],
                            tr[2], tr[6], tr[10], tr[14],
                            tr[3], tr[7], tr[11], tr[15]
                        );
                        DecomposeAffine(transform, sceneNode.translation, sceneNode.rotation, sceneNode.scale);
                    }
                    else
                    {
                        sceneNode.translation = gltfNode->has_translation ? float3(gltfNode->translation) : float3(0.0f, 0.0f, 0.0f);
                        sceneNode.rotation = gltfNode->has_rotation ? float4(gltfNode->rotation) : float4(0.0f, 0.0f, 0.0f, 1.0f);
                        sceneNode.scale = gltfNode->has_scale ? float3(gltfNode->scale) : float3(1.0f, 1.0f, 1.0f);
                    }

                    float4x4 translation;
                    translation.SetupByTranslation(sceneNode.translation);
                    float4x4 rotation;
                    rotation.SetupByQuaternion(sceneNode.rotation);
                    float4x4 scale;
                    scale.SetupByScale(sceneNode.scale);

                    sceneNode.localTransform = translation * (rotation * scale);

                    if (gltfNode->mesh)
                    {
                        sceneNode.instances = nodeToInstanceMap[gltfNode];
                        sceneNode.name = gltfNode->mesh->name ? gltfNode->mesh->name : ""; // TODO: gltfNode->name?
                    }
                }

                std::function<void(SceneNode*, SceneNode*)> setupGraphNodes = [&](SceneNode* parentNode, SceneNode* node)
                {
                    node->worldTransform = parentNode ? (parentNode->worldTransform * node->localTransform) : (scene.mSceneToWorld * node->localTransform);
                    node->parent = parentNode;

                    for (auto child : node->children)
                        setupGraphNodes(node, child);
                };

                for (cgltf_size nodeIndex = 0; nodeIndex < objects->scene->nodes_count; ++nodeIndex)
                {
                    uint32_t idx = (uint32_t)(objects->scene->nodes[nodeIndex] - objects->nodes);
                    setupGraphNodes(nullptr, &animation.sceneNodes[idx]);
                }
            }

            float animationTotalSec = 0.0f;
            for (uint32_t samplerIndex = 0; samplerIndex < gltfAnim->samplers_count; ++samplerIndex)
            {
                cgltf_animation_sampler* animSampler = gltfAnim->samplers + samplerIndex;
                float animTimeMaxSec = animSampler->input->has_max ? animSampler->input->max[0] : 0.0f;
                animationTotalSec = Max(animationTotalSec, animTimeMaxSec);
            }
            animation.animationTimeSec = animationTotalSec;
            animation.durationMs = animationTotalSec * 1000.0f;

            std::function<AnimationTrackType(cgltf_interpolation_type)> convertTrackType = [](cgltf_interpolation_type value)
            {
                switch (value)
                {
                    default:
                    case cgltf_interpolation_type_linear:
                        return AnimationTrackType::Linear;
                    case cgltf_interpolation_type_step:
                        return AnimationTrackType::Step;
                    case cgltf_interpolation_type_cubic_spline:
                        return AnimationTrackType::CubicSpline;
                }
            };

            for (uint32_t channelIndex = 0; channelIndex < gltfAnim->channels_count; ++channelIndex)
            {
                cgltf_animation_channel* animChannel = gltfAnim->channels + channelIndex;

                uint32_t index = (uint32_t)(animChannel->target_node - objects->nodes);
                SceneNode* sceneNode = animation.sceneNodes.data() + index;

                if (animChannel->sampler->input->count == 0)
                    continue;

                if (std::find(animation.dynamicNodes.begin(), animation.dynamicNodes.end(), sceneNode) == animation.dynamicNodes.end())
                    animation.dynamicNodes.push_back(sceneNode);

                switch (animChannel->target_path)
                {
                    case cgltf_animation_path_type_translation:
                    case cgltf_animation_path_type_scale:
                    {
                        bool isPosition = animChannel->target_path == cgltf_animation_path_type_translation;
                        if (isPosition)
                            animation.positionTracks.push_back(VectorAnimationTrack());
                        else
                            animation.scaleTracks.push_back(VectorAnimationTrack());

                        VectorAnimationTrack& track = isPosition ? animation.positionTracks.back() : animation.scaleTracks.back();
                        uint32_t frameCount = (uint32_t)animChannel->sampler->input->count;

                        track.node = sceneNode;
                        track.type = convertTrackType(animChannel->sampler->interpolation);
                        auto [keysSrc, keysStride] = cgltfBufferIterator(animChannel->sampler->input, sizeof(float));
                        auto [valuesSrc, valuesStride] = cgltfBufferIterator(animChannel->sampler->output, sizeof(float) * 3);
                        track.keys.reserve(frameCount);
                        track.values.reserve(frameCount);
                        for (uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex)
                        {
                            float key = *(float*)keysSrc;
                            keysSrc += keysStride;
                            track.keys.push_back(key);

                            float3 value = float3((float*)valuesSrc);
                            valuesSrc += valuesStride;
                            track.values.push_back(value);
                        }
                        track.frameCount = frameCount;
                    }
                    break;

                    case cgltf_animation_path_type_rotation:
                    {
                        animation.rotationTracks.push_back(QuatAnimationTrack());
                        QuatAnimationTrack& track = animation.rotationTracks.back();
                        uint32_t frameCount = (uint32_t)animChannel->sampler->input->count;

                        track.node = sceneNode;
                        track.type = convertTrackType(animChannel->sampler->interpolation);
                        auto [keysSrc, keysStride] = cgltfBufferIterator(animChannel->sampler->input, sizeof(float));
                        auto [valuesSrc, valuesStride] = cgltfBufferIterator(animChannel->sampler->output, sizeof(float) * 4);
                        track.keys.reserve(frameCount);
                        track.values.reserve(frameCount);
                        for (uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex)
                        {
                            float key = *(float*)keysSrc;
                            keysSrc += keysStride;
                            track.keys.push_back(key);

                            float4 value = float4((float*)valuesSrc);
                            valuesSrc += valuesStride;
                            track.values.push_back(value);
                        }
                        track.frameCount = frameCount;
                    }
                    break;

                    case cgltf_animation_path_type_weights:
                    {
                        uint32_t weightTrackIndex = (uint32_t)animation.weightTracks.size();
                        bool hasMeshInstance = false;
                        for (auto instanceIndex : sceneNode->instances)
                        {
                            Instance& instance = scene.instances[instanceIndex];
                            MeshInstance& meshInstance = scene.meshInstances[instance.meshInstanceIndex];
                            Mesh& mesh = scene.meshes[meshInstance.meshIndex];
                            if (mesh.HasMorphTargets())
                            {
                                // only take the first animation track 
                                if (std::find_if(animation.morphMeshInstances.begin(), animation.morphMeshInstances.end(), 
                                    [&instance](auto &x) { return x.meshInstanceIndex == instance.meshInstanceIndex; }) ==
                                    animation.morphMeshInstances.end())
                                {
                                    animation.morphMeshInstances.push_back({ weightTrackIndex, instance.meshInstanceIndex });
                                    hasMeshInstance = true;
                                }
                            }
                        }

                        if (hasMeshInstance)
                        {
                            animation.weightTracks.push_back(WeightsAnimationTrack());
                            WeightsAnimationTrack& track = animation.weightTracks.back();
                            uint32_t frameCount = (uint32_t)animChannel->sampler->input->count;
                            uint32_t outputCount = (uint32_t)animChannel->sampler->output->count;

                            track.type = convertTrackType(animChannel->sampler->interpolation);
                            auto [keysSrc, keysStride] = cgltfBufferIterator(animChannel->sampler->input, sizeof(float));
                            auto [valuesSrc, valuesStride] = cgltfBufferIterator(animChannel->sampler->output, sizeof(float));
                            track.keys.reserve(frameCount);
                            track.values.reserve(frameCount);
                            uint32_t numTargetsPerFrame = outputCount / frameCount;
                            track.frameCount = frameCount;

                            for (uint32_t frameIndex = 0; frameIndex < frameCount; ++frameIndex)
                            {
                                float key = *(float*)keysSrc;
                                keysSrc += keysStride;
                                track.keys.push_back(key);
                                track.values.push_back({});
                                auto& perFrameValues = track.values.back();
                                for (uint32_t targetIndex = 0; targetIndex < numTargetsPerFrame; targetIndex++)
                                {
                                    float weight = *(float*)valuesSrc;
                                    if (weight > 0.f)
                                    {
                                        perFrameValues.push_back(MorphTargetIndexWeight(targetIndex, weight));
                                    }
                                    valuesSrc += valuesStride;
                                }
                            }
                        }
                    }
                    break;
                }
            }
        }
    }

    uint32_t textureNum = (uint32_t)objects->textures_count;
    size_t newCapacity = scene.textures.size() + textureNum;
    scene.textures.reserve(newCapacity);

    if (scene.textures.empty())
    {
        // StaticTexture::Black
        {
            Texture* texture = new Texture;
            const std::string& texPath = GetFullPath("black.png", DataFolder::TEXTURES);
            NRI_ABORT_ON_FALSE(LoadTexture(texPath, *texture));
            scene.textures.push_back(texture);
        }

        // StaticTexture::White
        {
            Texture* texture = new Texture;
            const std::string& texPath = GetFullPath("white.png", DataFolder::TEXTURES);
            NRI_ABORT_ON_FALSE(LoadTexture(texPath, *texture));
            scene.textures.push_back(texture);
        }

        // StaticTexture::Invalid
        {
            Texture* texture = new Texture;
            const std::string& texPath = GetFullPath("checkerboard0.dds", DataFolder::TEXTURES);
            NRI_ABORT_ON_FALSE(LoadTexture(texPath, *texture, true));
            scene.textures.push_back(texture);
        }

        // StaticTexture::FlatNormal
        {
            Texture* texture = new Texture;
            const std::string& texPath = GetFullPath("flatnormal.png", DataFolder::TEXTURES);
            NRI_ABORT_ON_FALSE(LoadTexture(texPath, *texture));
            scene.textures.push_back(texture);
        }

        // StaticTexture::ScramblingRanking1spp
        {
            Texture* texture = new Texture;
            const std::string& texPath = GetFullPath("scrambling_ranking_128x128_2d_1spp.png", DataFolder::TEXTURES);
            NRI_ABORT_ON_FALSE(LoadTexture(texPath, *texture));
            texture->OverrideFormat(nri::Format::RGBA8_UINT);
            scene.textures.push_back(texture);
        }

        // StaticTexture::SobolSequence
        {
            Texture* texture = new Texture;
            const std::string& texPath = GetFullPath("sobol_256_4d.png", DataFolder::TEXTURES);
            NRI_ABORT_ON_FALSE(LoadTexture(texPath, *texture));
            texture->OverrideFormat(nri::Format::RGBA8_UINT);
            scene.textures.push_back(texture);
        }
    }

    // Materials
    std::unordered_map<const cgltf_image*, uint32_t> textures;
    for (uint32_t i = 0; i < materialNum; i++)
    {
        Material& material = scene.materials[materialOffset + i];

        const cgltf_material& gltfMaterial = objects->materials[i];

        uint32_t* textureIndices = &material.baseColorTexIndex;
        cgltf_texture* maps[4] = {nullptr};

        if (gltfMaterial.has_pbr_metallic_roughness)
        {
            maps[0] = gltfMaterial.pbr_metallic_roughness.base_color_texture.texture;
            maps[1] = gltfMaterial.pbr_metallic_roughness.metallic_roughness_texture.texture;
            material.baseColorAndMetalnessScale.x = gltfMaterial.pbr_metallic_roughness.base_color_factor[0];
            material.baseColorAndMetalnessScale.y = gltfMaterial.pbr_metallic_roughness.base_color_factor[1];
            material.baseColorAndMetalnessScale.z = gltfMaterial.pbr_metallic_roughness.base_color_factor[2];
            material.baseColorAndMetalnessScale.w = gltfMaterial.pbr_metallic_roughness.metallic_factor;
            material.emissiveAndRoughnessScale.w = gltfMaterial.pbr_metallic_roughness.roughness_factor;
            // TODO: opacity = gltfMaterial.pbr_metallic_roughness.base_color_factor[3]
        }
        else if (gltfMaterial.has_pbr_specular_glossiness)
        {
            // TODO: "pbr_specular_glossiness" model is not supported!
            maps[0] = gltfMaterial.pbr_specular_glossiness.diffuse_texture.texture;
            maps[1] = gltfMaterial.pbr_specular_glossiness.specular_glossiness_texture.texture;
        }


        bool useTransmission = false;
        if (gltfMaterial.has_transmission)
        {
            // TODO: use "gltfMaterial.transmission"
            useTransmission = true;
        }

        maps[2] = gltfMaterial.normal_texture.texture;
        // TODO: use "gltfMaterial.alpha_cutoff"?
        // TODO: use "gltfMaterial.double_sided"?

        maps[3] = gltfMaterial.emissive_texture.texture;
        material.emissiveAndRoughnessScale.x = gltfMaterial.emissive_factor[0];
        material.emissiveAndRoughnessScale.y = gltfMaterial.emissive_factor[1];
        material.emissiveAndRoughnessScale.z = gltfMaterial.emissive_factor[2];

        for (uint32_t j = 0; j < 4; j++)
        {
            cgltf_texture* texture = maps[j];
            if (!texture)
                continue;

            // See if the extensions include a DDS image
            const cgltf_image* ddsImage = ParseDdsImage(texture, objects);
            if ((!texture->image || (!texture->image->uri && !texture->image->buffer_view)) && (!ddsImage || (!ddsImage->uri && !ddsImage->buffer_view)))
                continue;

            // Pick either DDS or standard image, prefer DDS
            const cgltf_image* activeImage = (ddsImage && (ddsImage->uri || ddsImage->buffer_view)) ? ddsImage : texture->image;

            // Load a texture if not already loaded
            uint32_t textureIndex = 0;

            auto it = textures.find(activeImage);
            if (it == textures.end())
            {
                Texture* tex = new Texture;

                const bool computeAlphaMode = j == 0;
                const bool makeSRGB = j != 2;

                bool isLoaded = false;
                if (activeImage->buffer_view)
                {
                    assert(activeImage->buffer_view->size < std::numeric_limits<int>::max());

                    const uint8_t* data = ((const uint8_t*)activeImage->buffer_view->buffer->data) + activeImage->buffer_view->offset;
                    isLoaded = LoadTextureFromMemory(std::string(activeImage->name), data,
                        (int)activeImage->buffer_view->size, *tex, computeAlphaMode);
                }
                else
                {
                    std::string filename = (normPath.parent_path() / activeImage->uri).string();
                    isLoaded = LoadTexture(filename, *tex, computeAlphaMode);
                    if (!isLoaded)
                    {
                        std::string filenameDDS = filename.substr(0, filename.find_last_of('.')) + ".dds";
                        isLoaded = LoadTexture(filenameDDS, *tex, computeAlphaMode);
                    }   
                }

                if (isLoaded)
                {
                    if (makeSRGB)
                        tex->OverrideFormat(MakeSRGBFormat(tex->format));

                    textureIndex = (uint32_t)scene.textures.size();
                    scene.textures.push_back(tex);

                    textures[activeImage] = textureIndex;
                }
                else
                    delete tex;
                
                
               
            }
            else
                textureIndex = it->second;

            textureIndices[j] = textureIndex;
        }

        if (material.emissiveTexIndex == StaticTexture::Black && (material.emissiveAndRoughnessScale.x != 0.0f || material.emissiveAndRoughnessScale.y != 0.0f || material.emissiveAndRoughnessScale.z != 0.0f))
            material.emissiveTexIndex = StaticTexture::White;

        if (material.baseColorTexIndex == StaticTexture::Black && (material.baseColorAndMetalnessScale.x != 0.0f || material.baseColorAndMetalnessScale.y != 0.0f || material.baseColorAndMetalnessScale.z != 0.0f))
            material.baseColorTexIndex = StaticTexture::White;

        if (material.roughnessMetalnessTexIndex == StaticTexture::Black && (material.emissiveAndRoughnessScale.w != 0.0f || material.baseColorAndMetalnessScale.w != 0.0f))
            material.roughnessMetalnessTexIndex = StaticTexture::White;

        const Texture* diffuseTexture = scene.textures[material.baseColorTexIndex];
        material.alphaMode = useTransmission ? AlphaMode::TRANSPARENT : diffuseTexture->alphaMode;
        material.isHair = strstr(gltfMaterial.name, "hair") != 0;
        
        // TODO: remove strange polygon on the window in Kitchen scene
        if (strstr(gltfMaterial.name, "Material_295"))
            material.alphaMode = AlphaMode::OFF;

        /*
        switch (gltfMaterial.alpha_mode)
        {
        case cgltf_alpha_mode_opaque:
            material.alphaMode = useTransmission ? AlphaMode::OPAQUE : AlphaMode::OPAQUE;
            break;
        case cgltf_alpha_mode_mask:
            material.alphaMode = useTransmission ? AlphaMode::PREMULTIPLIED : AlphaMode::PREMULTIPLIED;
            break;
        case cgltf_alpha_mode_blend:
            material.alphaMode = useTransmission ? AlphaMode::TRANSPARENT : AlphaMode::TRANSPARENT;
            break;
        }
        */
    }

    // Set "Instance::allowUpdate" state
    std::function<void(SceneNode*)> setAllowUpdate = [&](SceneNode* sceneNode)
    {
        for (auto instanceIndex : sceneNode->instances)
        {
            Instance& instance = scene.instances[instanceIndex];
            instance.allowUpdate = true;
        }

        for (auto child : sceneNode->children)
            setAllowUpdate(child);
    };

    for (Animation& animation : scene.animations)
    {
        for (auto node : animation.dynamicNodes)
            setAllowUpdate(node);
    }

    // Cleanup
    cgltf_free(objects);

    return true;
}

void utils::Scene::Animate(float animationSpeed, float elapsedTime, float& animationProgress, uint32_t animationIndex)
{
    Animation& animation = animations[animationIndex];

    // Time
    float animationDelta = animation.durationMs == 0.0f ? 0.0f : animationSpeed / animation.durationMs;

    float t = animationProgress * 0.01f + elapsedTime * animationDelta * animation.sign;
    if (t >= 1.0f || t < 0.0f)
    {
        animation.sign = -animation.sign;
        t = Saturate(t);
    }

    animationProgress = t * 100.0f;

    float animTimeSec = t * animation.animationTimeSec;

    std::function<uint32_t(std::vector<float>&, float)> findKeyIndex = [](std::vector<float>& keys, float time)
    {
        if (time <= keys[0])
            return (uint32_t)0;

        if (time >= keys.back())
            return (uint32_t)(keys.size() - 1);

        for (int32_t index = (int32_t)keys.size() - 1; index >= 1; --index)
        {
            if (time >= keys[index])
                return (uint32_t)index;
        }

        return (uint32_t)0;
    };

    for (auto& track : animation.weightTracks)
    {
        track.activeValues.clear();

        uint32_t from = findKeyIndex(track.keys, animTimeSec);
        uint32_t to = Min(track.frameCount - 1, from + 1);
        float keyFrom = track.keys[from];
        float keyTo = track.keys[to];
        float time = animTimeSec < keyFrom ? keyFrom : (animTimeSec > keyTo ? keyTo : animTimeSec);
        float factor = to != from ? (time - keyFrom) / (keyTo - keyFrom) : 0.0f;
       
        switch (track.type)
        {
            case AnimationTrackType::Step:
            {
                track.activeValues = track.values[from];
            }
            break;

            case AnimationTrackType::CubicSpline: //TODO implement CubicSpline
            case AnimationTrackType::Linear:
            {
                const auto& morphsFrom = track.values[from];
                const auto& morphsTo = track.values[to];

                // morphsFrom and morphsTo are pre-sorted by morph target id
                // do a merge operation to interpolate shared target ids
                // if a target id doesn't exist in a key, it means it's weight is 0
                uint32_t fromIndex = 0;
                uint32_t toIndex = 0;

                float totalWeight = 0.f;
                while (fromIndex < morphsFrom.size() || toIndex < morphsTo.size())
                {
                    float fromWeight = 0.f;
                    float toWeight = 0.f;
                    uint32_t fromTargetId = ~0x0u;
                    uint32_t toTargetId = ~0x0u;

                    if (fromIndex < morphsFrom.size())
                    {
                        fromTargetId = morphsFrom[fromIndex].first;
                        fromWeight = morphsFrom[fromIndex].second;
                    }
                    if (toIndex < morphsTo.size())
                    {
                        toTargetId = morphsTo[toIndex].first;
                        toWeight = morphsTo[toIndex].second;
                    }

                    if (fromTargetId < toTargetId)
                    {
                        float interpWeight = Lerp(fromWeight, 0.f, factor);
                        totalWeight += interpWeight;
                        track.activeValues.emplace_back(fromTargetId, interpWeight);
                        fromIndex++;
                    }
                    else if (toTargetId < fromTargetId)
                    {
                        float interpWeight = Lerp(0.f, toWeight, factor);
                        totalWeight += interpWeight;
                        track.activeValues.emplace_back(toTargetId, interpWeight);

                        toIndex++;
                    }
                    else //if (fromTargetId == toTargetId)
                    {
                        float interpWeight = Lerp(fromWeight, toWeight, factor);
                        totalWeight += interpWeight;
                        track.activeValues.emplace_back(fromTargetId, interpWeight);
                        fromIndex++;
                        toIndex++;
                    }
                }
                // sort by weight descending
                std::sort(track.activeValues.begin(), track.activeValues.end(), [](auto& elemA, auto& elemB) { return elemA.second >= elemB.second; });

                if (totalWeight != 1.0f)
                {
                    // renormalize
                    float totalWeightRcp = 1.0f / totalWeight;
                    for (auto& activeMorphValue : track.activeValues) {
                        activeMorphValue.second *= totalWeightRcp;
                    }
                }
            }
            break;
        }
    }

    for (auto& track : animation.positionTracks)
    {
        uint32_t from = findKeyIndex(track.keys, animTimeSec);
        uint32_t to = Min(track.frameCount - 1, from + 1);
        float keyFrom = track.keys[from];
        float keyTo = track.keys[to];
        float time = animTimeSec < keyFrom ? keyFrom : (animTimeSec > keyTo ? keyTo : animTimeSec);
        float factor = to != from ? (time - keyFrom) / (keyTo - keyFrom) : 0.0f;
        float3 value = float3(0.0f, 0.0f, 0.0f);

        switch (track.type)
        {
            case AnimationTrackType::Step:
            {
                value = track.values[from];
            }
            break;

            case AnimationTrackType::CubicSpline: //TODO implement CubicSpline
            case AnimationTrackType::Linear:
            {
                value = Lerp(track.values[from], track.values[to], float3(factor));
            }
            break;
        }

        track.node->translation = value;
    }

    for (auto& track : animation.rotationTracks)
    {
        uint32_t from = findKeyIndex(track.keys, animTimeSec);
        uint32_t to = Min(track.frameCount - 1, from + 1);
        float keyFrom = track.keys[from];
        float keyTo = track.keys[to];
        float time = animTimeSec < keyFrom ? keyFrom : (animTimeSec > keyTo ? keyTo : animTimeSec);
        float factor = to != from ? (time - keyFrom) / (keyTo - keyFrom) : 0.0f;
        float4 value = float4(0.0f, 0.0f, 0.0f, 1.0f);

        switch (track.type)
        {
            case AnimationTrackType::Step:
            {
                value = track.values[from];
            }
            break;

            case AnimationTrackType::CubicSpline: //TODO implement CubicSpline
            case AnimationTrackType::Linear:
            {
                float4 a = track.values[from];
                float4 b = track.values[to];
                float theta = Dot44(a, b);
                a = (theta < 0.0f) ? -a : a;
                value = Slerp(a, b, factor);
            }
            break;
        }

        track.node->rotation = value;
    }

    for (auto& track : animation.scaleTracks)
    {
        uint32_t from = findKeyIndex(track.keys, animTimeSec);
        uint32_t to = Min(track.frameCount - 1, from + 1);
        float keyFrom = track.keys[from];
        float keyTo = track.keys[to];
        float time = animTimeSec < keyFrom ? keyFrom : (animTimeSec > keyTo ? keyTo : animTimeSec);
        float factor = to != from ? (time - keyFrom) / (keyTo - keyFrom) : 0.0f;
        float3 value = float3(1.0f, 1.0f, 1.0f);

        switch (track.type)
        {
            case AnimationTrackType::Step:
                value = track.values[from];
                break;

            case AnimationTrackType::CubicSpline: //TODO implement CubicSpline
            case AnimationTrackType::Linear:
                value = Lerp(track.values[from], track.values[to], float3(factor));
                break;
        }

        track.node->scale = value;
    }

    std::function<void(SceneNode*)> updateChain = [&](SceneNode* sceneNode)
    {
        float4x4 translation;
        translation.SetupByTranslation(sceneNode->translation);
        float4x4 rotation;
        rotation.SetupByQuaternion(sceneNode->rotation);
        float4x4 scale;
        scale.SetupByScale(sceneNode->scale);

        sceneNode->localTransform = translation * (rotation * scale);
        sceneNode->worldTransform = sceneNode->parent ? (sceneNode->parent->worldTransform * sceneNode->localTransform) : (mSceneToWorld * sceneNode->localTransform);

        float4x4 transform = sceneNode->worldTransform;
        double3 position = ToDouble(transform.GetCol3().To3d());
        transform.SetTranslation(float3::Zero());

        for (auto instanceIndex : sceneNode->instances)
        {
            Instance& instance = this->instances[instanceIndex];
            instance.rotation = transform;
            instance.position = position;
        }

        for (auto child : sceneNode->children)
            updateChain(child);
    };

    // TODO: this could be optimized by only updating roots of the dynamic chains, i.e. when one dynamic node is hierarchical child of another
    for (auto node : animation.dynamicNodes)
        updateChain(node);
}
