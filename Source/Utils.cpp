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

#include "assimp/scene.h"
#include "assimp/cimport.h"
#include "assimp/postprocess.h"

#include "Detex/detex.h"

#include "MathLib/MathLib.h"

#include "NRI.h"
#include "Extensions/NRIHelper.h"

#include "Helper.h"
#include "Utils.h"

constexpr std::array<aiTextureType, 5> gSupportedTextureTypes =
{
    aiTextureType_DIFFUSE,      // OBJ - map_Kd
    aiTextureType_SPECULAR,     // OBJ - map_Ks
    aiTextureType_NORMALS,      // OBJ - map_Kn
    aiTextureType_EMISSIVE,     // OBJ - map_Ke
    aiTextureType_SHININESS     // OBJ - map_Ns (smoothness)
};

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

inline bool IsExist(const std::string& path)
{
#if _WIN32
    return _access(path.c_str(), 0) == 0;
#else
    return access(path.c_str(), 0) == 0;
#endif
}

inline bool EndsWithNoCase(std::string const &value, std::string const &ending)
{
    auto it = ending.begin();
    return value.size() >= ending.size() &&
        std::all_of(std::next(value.begin(), value.size() - ending.size()), value.end(), [&it](const char & c) {
            return ::tolower(c) == ::tolower(*(it++));
        });
}

static void GenerateTangents(const aiMesh& mesh, std::vector<float4>& tangents)
{
    std::vector<float3> tan1(mesh.mNumVertices, float3::Zero());
    std::vector<float3> tan2(mesh.mNumVertices, float3::Zero());

    const aiVector3D zeroUv(0.0f, 0.0f, 0.0f);
    const bool hasTexCoord0 = mesh.HasTextureCoords(0);

    for (uint32_t i = 0; i < mesh.mNumFaces; i++)
    {
        uint32_t i0 = mesh.mFaces[i].mIndices[0];
        uint32_t i1 = mesh.mFaces[i].mIndices[1];
        uint32_t i2 = mesh.mFaces[i].mIndices[2];

        float3 p0 = float3(&mesh.mVertices[i0].x);
        float3 p1 = float3(&mesh.mVertices[i1].x);
        float3 p2 = float3(&mesh.mVertices[i2].x);

        float3 n0 = float3(&mesh.mNormals[i0].x);
        float3 n1 = float3(&mesh.mNormals[i1].x);
        float3 n2 = float3(&mesh.mNormals[i2].x);

        const aiVector3D& uv0 = hasTexCoord0 ? mesh.mTextureCoords[0][i0] : zeroUv;
        const aiVector3D& uv1 = hasTexCoord0 ? mesh.mTextureCoords[0][i1] : zeroUv;
        const aiVector3D& uv2 = hasTexCoord0 ? mesh.mTextureCoords[0][i2] : zeroUv;

        float s1 = uv1.x - uv0.x;
        float s2 = uv2.x - uv0.x;
        float t1 = uv1.y - uv0.y;
        float t2 = uv2.y - uv0.y;
        float r = s1 * t2 - s2 * t1;

        float3 sdir, tdir;

        if (Abs(r) < 1e-9f)
        {
            n1.z += 1e-6f;
            sdir = GetPerpendicularVector(n1);
            tdir = Cross(n1, sdir);
        }
        else
        {
            float invr = 1.0f / r;

            float3 a = (p1 - p0) * invr;
            float3 b = (p2 - p0) * invr;

            sdir = a * t2 - b * t1;
            tdir = b * s1 - a * s2;
        }

        tan1[i0] += sdir;
        tan1[i1] += sdir;
        tan1[i2] += sdir;

        tan2[i0] += tdir;
        tan2[i1] += tdir;
        tan2[i2] += tdir;
    }

    for (uint32_t i = 0; i < mesh.mNumVertices; i++)
    {
        float3 n = float3(&mesh.mNormals[i].x);

        float3 t = tan1[i];
        if (t.IsZero())
           t = Cross(tan2[i], n);

        // Gram-Schmidt orthogonalize
        t -= n * Dot33(n, t);
        float len = Length(t);
        t /= Max(len, 1e-9f);

        // Calculate handedness
        float handedness = Sign( Dot33( Cross(n, t), tan2[i] ) );

        tangents[i] = float4(t.x, t.y, t.z, handedness);
    }
}

inline uint64_t ComputeHash(const void* key, uint32_t len)
{
    const uint8_t* p = (uint8_t*)key;
    uint64_t result = 14695981039346656037ull;
    while( len-- )
        result = (result ^ (*p++)) * 1099511628211ull;

    return result;
}

inline const char* GetShaderExt(nri::GraphicsAPI graphicsAPI)
{
    if (graphicsAPI == nri::GraphicsAPI::D3D11)
        return ".dxbc";
    else if (graphicsAPI == nri::GraphicsAPI::D3D12)
        return ".dxil";

    return ".spirv";
}

struct NodeData
{
    aiMatrix4x4 transform;
    aiNode* node;
};

static void BindNodesToMeshID(aiNode* node, aiMatrix4x4 parentTransform, std::vector<std::vector<NodeData>>& vector)
{
    aiMatrix4x4 transform = parentTransform * node->mTransformation;

    for (uint32_t j = 0; j < node->mNumMeshes; j++)
    {
        NodeData tmpNodeData = {};
        tmpNodeData.node = node;
        tmpNodeData.transform = transform;
        vector[node->mMeshes[j]].push_back(tmpNodeData);
    }

    for (uint32_t i = 0; i < node->mNumChildren; i++)
        BindNodesToMeshID(node->mChildren[i], transform, vector);
}

static void ExtractNodeTree(const aiNode* node, std::map<const aiNode*, std::vector<uint32_t>>& nodeToInstanceMap, utils::NodeTree& animationInstance)
{
    float4x4 transform = float4x4(
        node->mTransformation.a1, node->mTransformation.a2, node->mTransformation.a3, node->mTransformation.a4,
        node->mTransformation.b1, node->mTransformation.b2, node->mTransformation.b3, node->mTransformation.b4,
        node->mTransformation.c1, node->mTransformation.c2, node->mTransformation.c3, node->mTransformation.c4,
        0.0f, 0.0f, 0.0f, 1.0f);

    utils::NodeTree parentInstance = {};
    parentInstance.hash = ComputeHash((uint8_t*)node->mName.C_Str(), node->mName.length);
    parentInstance.mTransform = transform;

    std::map<const aiNode*, std::vector<uint32_t>>::iterator it = nodeToInstanceMap.find((aiNode*)node);
    if (it != nodeToInstanceMap.end())
        parentInstance.instances = it->second;

    parentInstance.children.resize(node->mNumChildren);
    for (uint32_t i = 0; i < node->mNumChildren; i++)
        ExtractNodeTree(node->mChildren[i], nodeToInstanceMap, parentInstance.children[i]);

    animationInstance = parentInstance;
}

inline float SafeLinearstep(float a, float b, float x)
{
    return a == b ? 0.0f : Linearstep(a, b, x);
}

inline uint32_t FindCurrentIndex(const std::vector<float>& keys, float time)
{
    for (uint32_t i = helper::GetCountOf(keys) - 1; i >= 1; i--)
    {
        if (time >= keys[i])
            return i;
    }

    return 0u;
};

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

    texture.mips = (Mip*)dTexture;
    texture.name = path;
    texture.hash = ComputeHash(path.c_str(), (uint32_t)path.length());
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
        detexTexture *lastMip = dTexture[mipNum - 1];
        uint8_t *rgba8 = lastMip->data;
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

        if (texture.alphaMode != AlphaMode::PREMULTIPLIED && avgColor.w < 254.5f / 255.0f)
            texture.alphaMode = AlphaMode::TRANSPARENT;

        if (texture.alphaMode == AlphaMode::TRANSPARENT && avgColor.w == 0.0f)
        {
            printf("WARNING: Texture '%s' is fully transparent!\n", path.c_str());
            texture.alphaMode = AlphaMode::OFF;
        }
    }

    return true;
}

void utils::LoadTextureFromMemory(nri::Format format, uint32_t width, uint32_t height, const uint8_t *pixels, Texture &texture)
{
    detexTexture **dTexture;
    detexLoadTextureFromMemory(DETEX_PIXEL_FORMAT_RGBA8, width, height, pixels, &dTexture);

    texture.mipNum = 1;
    texture.arraySize = 1;
    texture.depth = 1;
    texture.format = format;
    texture.alphaMode = AlphaMode::OPAQUE;
    texture.mips = (Mip*)dTexture;
}

bool utils::LoadScene(const std::string& path, Scene& scene, bool allowUpdate)
{
    printf("Loading scene '%s'...\n", GetFileName(path));

    constexpr uint32_t MAX_INDEX = 65535;

    // Taken from Falcor
    uint32_t aiFlags = aiProcessPreset_TargetRealtime_MaxQuality;
    aiFlags |= aiProcess_FlipUVs;
    aiFlags &= ~aiProcess_CalcTangentSpace; // Use Mikktspace instead
    aiFlags &= ~aiProcess_FindDegenerates; // Avoid converting degenerated triangles to lines
    aiFlags &= ~aiProcess_OptimizeGraph; // Never use as it doesn't handle transforms with negative determinants
    aiFlags &= ~aiProcess_OptimizeMeshes; // Avoid merging original meshes

    std::string baseDir = path;
    const size_t lastSlash = baseDir.find_last_of("\\/");
    if (lastSlash != std::string::npos)
        baseDir.erase(baseDir.begin() + lastSlash + 1, baseDir.end());

    aiPropertyStore* props = aiCreatePropertyStore();
    aiSetImportPropertyInteger(props, AI_CONFIG_PP_SLM_VERTEX_LIMIT, MAX_INDEX);
    aiSetImportPropertyInteger(props, AI_CONFIG_PP_RVC_FLAGS, aiComponent_COLORS);
    aiSetImportPropertyInteger(props, AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_LINE | aiPrimitiveType_POINT);
    const aiScene* result = aiImportFileExWithProperties(path.c_str(), aiFlags, nullptr, props);
    aiReleasePropertyStore(props);
    if (!result)
    {
        printf("ERROR: Can't load scene '%s'...\n", path.c_str());
        return false;
    }
    const aiScene& aiScene = *result;
    const aiNode* rootNode = aiScene.mRootNode;

    uint32_t instanceNum = aiScene.mNumMeshes;
    std::map<const aiNode*, std::vector<uint32_t>> nodeToInstanceMap;
    std::vector<std::vector<NodeData>> nodesByMeshID;
    const bool loadFromNodes = EndsWithNoCase(path.c_str(), ".fbx") != 0 || EndsWithNoCase(path.c_str(), ".gltf") != 0;
    if (loadFromNodes)
    {
        nodesByMeshID.resize(aiScene.mNumMeshes);
        for (uint32_t j = 0; j < rootNode->mNumChildren; j++)
            BindNodesToMeshID(rootNode->mChildren[j], rootNode->mTransformation, nodesByMeshID);

        instanceNum = 0;
        for (uint32_t j = 0; j < helper::GetCountOf(nodesByMeshID); j++)
            instanceNum += helper::GetCountOf(nodesByMeshID[j]);

        scene.mSceneToWorld.SetupByRotationX( Pi(0.5f) );
    }

    // Group meshes by material
    std::vector<std::pair<uint32_t, uint32_t>> sortedMaterials(aiScene.mNumMeshes);
    {
        for (uint32_t i = 0; i < aiScene.mNumMeshes; i++)
            sortedMaterials[i] = { i, aiScene.mMeshes[i]->mMaterialIndex };

        const auto sortPred = [](const std::pair<uint32_t, uint32_t>& a, const std::pair<uint32_t, uint32_t>& b)
        { return a.second < b.second; };

        std::sort(sortedMaterials.begin(), sortedMaterials.end(), sortPred);
    }

    const uint32_t materialOffset = (uint32_t)scene.materials.size();
    scene.materials.resize(materialOffset + aiScene.mNumMaterials);

    // Meshes and instances
    const uint32_t meshOffset = (uint32_t)scene.meshes.size();
    scene.meshes.resize(meshOffset + aiScene.mNumMeshes);

    const uint32_t instanceOffset = (uint32_t)scene.instances.size();
    scene.instances.resize(instanceOffset + instanceNum);

    uint32_t totalIndices = (uint32_t)scene.indices.size();
    uint32_t totalVertices = (uint32_t)scene.vertices.size();
    uint32_t indexOffset = totalIndices;
    uint32_t vertexOffset = totalVertices;

    uint32_t nodeInstanceOffset = instanceOffset;
    for (uint32_t i = 0; i < aiScene.mNumMeshes; i++)
    {
        const uint32_t sortedMeshIndex = sortedMaterials[i].first;
        const aiMesh& aiMesh = *aiScene.mMeshes[sortedMeshIndex];
        const uint32_t meshIndex = meshOffset + i;

        Mesh& mesh = scene.meshes[meshIndex];
        mesh.indexOffset = totalIndices;
        mesh.vertexOffset = totalVertices;
        mesh.indexNum = aiMesh.mNumFaces * 3;
        mesh.vertexNum = aiMesh.mNumVertices;

        totalIndices += mesh.indexNum;
        totalVertices += mesh.vertexNum;

        uint32_t materialIndex = materialOffset + sortedMaterials[i].second;
        if (loadFromNodes)
        {
            const std::vector<NodeData>& relatedNodes = nodesByMeshID[sortedMeshIndex];

            for (uint32_t j = 0; j < helper::GetCountOf(relatedNodes); j++)
            {
                const aiMatrix4x4& aiTransform = relatedNodes[j].transform;

                float4x4 transform
                (
                    aiTransform.a1, aiTransform.a2, aiTransform.a3, aiTransform.a4,
                    aiTransform.b1, aiTransform.b2, aiTransform.b3, aiTransform.b4,
                    aiTransform.c1, aiTransform.c2, aiTransform.c3, aiTransform.c4,
                    0.0f, 0.0f, 0.0f, 1.0f
                );
                transform = scene.mSceneToWorld * transform;

                double3 position = ToDouble( transform.GetCol3().To3d() );
                transform.SetTranslation( float3::Zero() );

                Instance& instance = scene.instances[nodeInstanceOffset];
                instance.meshIndex = meshIndex;
                instance.position = position;
                instance.rotation = transform;
                instance.materialIndex = materialIndex;
                instance.allowUpdate = allowUpdate;

                std::map<const aiNode*, std::vector<uint32_t>>::iterator nodeToInstIt = nodeToInstanceMap.find(relatedNodes[j].node);
                if (nodeToInstIt != nodeToInstanceMap.end())
                    nodeToInstIt->second.push_back(nodeInstanceOffset);
                else
                {
                    std::vector<uint32_t> tmpVector = { nodeInstanceOffset };
                    nodeToInstanceMap.insert( std::make_pair(relatedNodes[j].node, tmpVector) );
                }

                nodeInstanceOffset++;
            }
        }
        else
        {
            Instance& instance = scene.instances[instanceOffset + i];
            instance.meshIndex = meshIndex;
            instance.materialIndex = materialIndex;
            instance.allowUpdate = allowUpdate;
        }
    }

    // Animation
    if (aiScene.HasAnimations())
    {
        const aiAnimation* aiAnimation = aiScene.mAnimations[0];
        scene.animations.push_back(Animation());
        Animation& animation = scene.animations.back();
        animation.animationName = GetFileName(path);

        ExtractNodeTree(rootNode, nodeToInstanceMap, animation.rootNode);

        float animationTotalMs = float(1000.0 * aiAnimation->mDuration / aiAnimation->mTicksPerSecond);
        animation.durationMs = animationTotalMs;
        animation.animationNodes.resize(aiAnimation->mNumChannels);

        for(uint32_t i = 0; i < aiAnimation->mNumChannels; i++)
        {
            const aiNodeAnim* animChannel = aiAnimation->mChannels[i];
            const aiNode* affectedNode = rootNode->FindNode(animChannel->mNodeName);

            // Camera
            bool isCamera = false;
            if (aiScene.mNumCameras > 0 && strstr(animChannel->mNodeName.C_Str(), aiScene.mCameras[0]->mName.C_Str()))
            {
                NodeTree* nextNode = &animation.cameraNode;
                while (!nextNode->children.empty())
                    nextNode = &nextNode->children[0];

                nextNode->animationNodeIndex = i;
                nextNode->children.push_back( NodeTree() );

                isCamera = true;
            }

            // Objects
            const uint64_t hash = ComputeHash(affectedNode->mName.C_Str(), affectedNode->mName.length);
            AnimationNode& animationNode = animation.animationNodes[i];

            for (uint32_t j = 0; j < animChannel->mNumPositionKeys; j++)
            {
                const aiVectorKey& positionKey = animChannel->mPositionKeys[j];
                const float time = float( positionKey.mTime / aiAnimation->mDuration );
                const double3 value = double3(positionKey.mValue.x, positionKey.mValue.y, positionKey.mValue.z);
                animationNode.positionKeys.push_back(time);
                animationNode.positionValues.push_back(value);
            }

            for (uint32_t j = 0; j < animChannel->mNumRotationKeys; j++)
            {
                const aiQuatKey& rotationKey = animChannel->mRotationKeys[j];
                const float time = float( rotationKey.mTime / aiAnimation->mDuration );
                const float4 value = float4(rotationKey.mValue.x, rotationKey.mValue.y, rotationKey.mValue.z, rotationKey.mValue.w);
                animationNode.rotationKeys.push_back(time);
                animationNode.rotationValues.push_back(value);
            }

            for (uint32_t j = 0; j < animChannel->mNumScalingKeys; j++)
            {
                const aiVectorKey& scalingKey = animChannel->mScalingKeys[j];
                const float time = float( scalingKey.mTime / aiAnimation->mDuration );
                const float3 value = float3(scalingKey.mValue.x, scalingKey.mValue.y, scalingKey.mValue.z);
                animationNode.scaleKeys.push_back(time);
                animationNode.scaleValues.push_back(value);
            }

            std::function<void(const uint64_t&, NodeTree&)> findNode = [&](const uint64_t& hash, NodeTree& sceneNode)
            {
                if (hash == sceneNode.hash)
                {
                    sceneNode.animationNodeIndex = i;
                    return;
                }

                for (NodeTree& child : sceneNode.children)
                    findNode(hash, child);
            };

            findNode(hash, animation.rootNode);
        }
    }

    // Geometry
    std::vector<float4> tangents(totalVertices);

    scene.indices.resize(totalIndices);
    scene.primitives.resize(totalIndices / 3);
    scene.vertices.resize(totalVertices);
    scene.unpackedVertices.resize(totalVertices);

    for (uint32_t i = 0; i < aiScene.mNumMeshes; i++)
    {
        const uint32_t sortedMeshIndex = sortedMaterials[i].first;
        const aiMesh& aiMesh = *aiScene.mMeshes[sortedMeshIndex];

        Mesh& mesh = scene.meshes[meshOffset + i];
        mesh.aabb.Clear();

        // Generate tangents
        GenerateTangents(aiMesh, tangents);

        // Indices
        for (uint32_t j = 0; j < aiMesh.mNumFaces; j++)
        {
            const aiFace& aiFace = aiMesh.mFaces[j];
            for (uint32_t k = 0; k < aiFace.mNumIndices; k++)
            {
                uint32_t index = aiFace.mIndices[k];
                scene.indices[indexOffset++] = (uint16_t)index;
            }
        }

        // Vertices
        for (uint32_t j = 0; j < aiMesh.mNumVertices; j++)
        {
            UnpackedVertex& unpackedVertex = scene.unpackedVertices[vertexOffset];
            Vertex& vertex = scene.vertices[vertexOffset++];

            // Position
            float3 position = float3(&aiMesh.mVertices[j].x);
            vertex.position[0] = position.x;
            vertex.position[1] = position.y;
            vertex.position[2] = position.z;
            unpackedVertex.position[0] = position.x;
            unpackedVertex.position[1] = position.y;
            unpackedVertex.position[2] = position.z;
            mesh.aabb.Add(position);

            // Normal
            float3 normal = -float3(&aiMesh.mNormals[j].x); // TODO: why negated?
            if( All<CmpLess>(Abs(normal), 1e-6f) )
                normal = float3(0.0f, 0.0f, 1.0f); // zero vector

            unpackedVertex.normal[0] = normal.x;
            unpackedVertex.normal[1] = normal.y;
            unpackedVertex.normal[2] = normal.z;
            vertex.normal = Packed::uf4_to_uint<10, 10, 10, 2>(normal * 0.5f + 0.5f);

            // Tangent
            float4 tangent = tangents[j];
            if( All<CmpLess>(Abs(float3(tangent.xmm)), 1e-6f) )
                tangent = float4(0.0f, 0.0f, 1.0f, 1.0f); // zero vector

            unpackedVertex.tangent[0] = tangent.x;
            unpackedVertex.tangent[1] = tangent.y;
            unpackedVertex.tangent[2] = tangent.z;
            unpackedVertex.tangent[3] = tangent.w;
            vertex.tangent = Packed::uf4_to_uint<10, 10, 10, 2>(tangent * 0.5f + 0.5f);

            // Uv
            if (aiMesh.HasTextureCoords(0))
            {
                float u = Min( aiMesh.mTextureCoords[0][j].x, 65504.0f );
                float v = Min( aiMesh.mTextureCoords[0][j].y, 65504.0f );

                unpackedVertex.uv[0] = u;
                unpackedVertex.uv[1] = v;
                vertex.uv = Packed::sf2_to_h2(u, v);
            }
            else
            {
                unpackedVertex.uv[0] = 0.0f;
                unpackedVertex.uv[1] = 0.0f;
                vertex.uv = 0;
            }
        }

        // Primitive data
        uint32_t triangleNum = mesh.indexNum / 3;
        for (uint32_t j = 0; j < triangleNum; j++)
        {
            uint32_t primitiveIndex = mesh.indexOffset / 3 + j;
            const UnpackedVertex& v0 = scene.unpackedVertices[ mesh.vertexOffset + scene.indices[primitiveIndex * 3] ];
            const UnpackedVertex& v1 = scene.unpackedVertices[ mesh.vertexOffset + scene.indices[primitiveIndex * 3 + 1] ];
            const UnpackedVertex& v2 = scene.unpackedVertices[ mesh.vertexOffset + scene.indices[primitiveIndex * 3 + 2] ];

            float3 p0(v0.position);
            float3 p1(v1.position);
            float3 p2(v2.position);

            float3 edge20 = p2 - p0;
            float3 edge10 = p1 - p0;
            float worldArea = Max( Length( Cross(edge20, edge10) ), 1e-9f );

            float3 uvEdge20 = float3(v2.uv[0], v2.uv[1], 0.0f) - float3(v0.uv[0], v0.uv[1], 0.0f);
            float3 uvEdge10 = float3(v1.uv[0], v1.uv[1], 0.0f) - float3(v0.uv[0], v0.uv[1], 0.0f);
            float uvArea = Length( Cross(uvEdge20, uvEdge10) );

            Primitive& primitive = scene.primitives[primitiveIndex];
            primitive.worldToUvUnits = uvArea == 0 ? 1.0f : Sqrt( uvArea / worldArea );

            // Unsigned curvature
            // https://computergraphics.stackexchange.com/questions/1718/what-is-the-simplest-way-to-compute-principal-curvature-for-a-mesh-triangle
            float3 n0 = float3(v0.normal);
            float3 n1 = float3(v1.normal);
            float3 n2 = float3(v2.normal);

            // Stage 1
            float curvature10 = Abs( Dot33(n1 - n0, p1 - p0) ) / LengthSquared(p1 - p0);
            float curvature21 = Abs( Dot33(n2 - n1, p2 - p1) ) / LengthSquared(p2 - p1);
            float curvature02 = Abs( Dot33(n0 - n2, p0 - p2) ) / LengthSquared(p0 - p2);

            primitive.curvature = Max(curvature10, curvature21);
            primitive.curvature = Max(primitive.curvature, curvature02);

            // Stage 2
            float invTriArea = 1.0f / (worldArea * 0.5f);
            curvature10 = Sqrt( LengthSquared(n1 - n0) * invTriArea );
            curvature21 = Sqrt( LengthSquared(n2 - n1) * invTriArea );
            curvature02 = Sqrt( LengthSquared(n0 - n2) * invTriArea );

            primitive.curvature = Max(primitive.curvature, curvature10);
            primitive.curvature = Max(primitive.curvature, curvature21);
            primitive.curvature = Max(primitive.curvature, curvature02);
        }

        // Scene AABB
        if (loadFromNodes)
        {
            for (const auto& instance : scene.instances)
            {
                if (instance.meshIndex == meshOffset + i)
                {
                    float4x4 transform = instance.rotation;
                    transform.AddTranslation( ToFloat(instance.position) );

                    cBoxf aabb;
                    TransformAabb(transform, mesh.aabb, aabb);

                    scene.aabb.Add(aabb);
                }
            }
        }
        else
            scene.aabb.Add(mesh.aabb);
    }

    // Count textures
    aiString str;
    uint32_t textureNum = 0;
    for (uint32_t i = 0; i < aiScene.mNumMaterials; i++)
    {
        const aiMaterial* assimpMaterial = aiScene.mMaterials[i];
        for (size_t j = 0; j < gSupportedTextureTypes.size(); j++)
        {
            if (assimpMaterial->GetTexture(gSupportedTextureTypes[j], 0, &str) == AI_SUCCESS)
                textureNum++;
        }
    }

    size_t newCapacity = scene.textures.size() + textureNum;
    scene.textures.reserve(newCapacity);

    // StaticTexture::Black
    {
        Texture* texture = new Texture;
        const std::string& texPath = GetFullPath("black.png", DataFolder::TEXTURES);
        NRI_ABORT_ON_FALSE( LoadTexture(texPath, *texture) );
        scene.textures.push_back(texture);
    }

    // StaticTexture::Invalid
    {
        Texture* texture = new Texture;
        const std::string& texPath = GetFullPath("checkerboard0.dds", DataFolder::TEXTURES);
        NRI_ABORT_ON_FALSE( LoadTexture(texPath, *texture, true) );
        scene.textures.push_back(texture);
    }

    // StaticTexture::FlatNormal
    {
        Texture* texture = new Texture;
        const std::string& texPath = GetFullPath("flatnormal.png", DataFolder::TEXTURES);
        NRI_ABORT_ON_FALSE( LoadTexture(texPath, *texture) );
        scene.textures.push_back(texture);
    }

    // StaticTexture::ScramblingRanking1spp
    {
        Texture* texture = new Texture;
        const std::string& texPath = GetFullPath("scrambling_ranking_128x128_2d_1spp.png", DataFolder::TEXTURES);
        NRI_ABORT_ON_FALSE( LoadTexture(texPath, *texture) );
        texture->OverrideFormat(nri::Format::RGBA8_UINT);
        scene.textures.push_back(texture);
    }

    // StaticTexture::SobolSequence
    {
        Texture* texture = new Texture;
        const std::string& texPath = GetFullPath("sobol_256_4d.png", DataFolder::TEXTURES);
        NRI_ABORT_ON_FALSE( LoadTexture(texPath, *texture) );
        texture->OverrideFormat(nri::Format::RGBA8_UINT);
        scene.textures.push_back(texture);
    }

    // Load only unique textures
    for (uint32_t i = 0; i < aiScene.mNumMaterials; i++)
    {
        Material& material = scene.materials[materialOffset + i];
        uint32_t* textureIndices = &material.diffuseMapIndex;

        const aiMaterial* assimpMaterial = aiScene.mMaterials[i];
        for (size_t j = 0; j < gSupportedTextureTypes.size(); j++)
        {
            aiTextureType type = gSupportedTextureTypes[j];
            if (assimpMaterial->GetTexture(type, 0, &str) == AI_SUCCESS)
            {
                std::string texPath = baseDir + str.data;
                const uint64_t hash = ComputeHash(texPath.c_str(), (uint32_t)texPath.length());

                const auto comparePred = [&hash](const Texture* texture)
                { return hash == texture->hash; };

                auto findResult = std::find_if(scene.textures.begin(), scene.textures.end(), comparePred);
                if (findResult == scene.textures.end())
                {
                    const bool isMaterial = type == aiTextureType_DIFFUSE || type == aiTextureType_EMISSIVE || type == aiTextureType_SPECULAR;

                    Texture* texture = new Texture;
                    bool isLoaded = LoadTexture(texPath, *texture, isMaterial);

                    if (isLoaded)
                    {
                        if (isMaterial)
                            texture->OverrideFormat(MakeSRGBFormat(texture->format));

                        textureIndices[j] = (uint32_t)scene.textures.size();
                        scene.textures.push_back(texture);
                    }
                    else
                        delete texture;
                }
                else
                    textureIndices[j] = (uint32_t)(findResult - scene.textures.begin());
            }
        }

        const Texture* diffuseTexture = scene.textures[material.diffuseMapIndex];
        material.alphaMode = diffuseTexture->alphaMode;
    }

    // Cleanup
    aiReleaseImport(&aiScene);

    // TODO: some sorting can be added here (remapping is needed)

    // Set "Instance::allowUpdate" state
    for (Animation& animation : scene.animations)
        animation.rootNode.SetAllowUpdate(scene, allowUpdate);

    return true;
}

void utils::AnimationNode::Update(float time)
{
    float3 scale = scaleValues.back();
    if (time < scaleKeys.back())
    {
        uint32_t firstID = FindCurrentIndex(scaleKeys, time);
        uint32_t secondID = (firstID + 1) % scaleKeys.size();

        float weight = SafeLinearstep(scaleKeys[firstID], scaleKeys[secondID], time);
        scale = Lerp(scaleValues[firstID], scaleValues[secondID], float3(weight));
    }

    float4 rotation = rotationValues.back();
    if (time < rotationKeys.back())
    {
        uint32_t firstID = FindCurrentIndex(rotationKeys, time);
        uint32_t secondID = (firstID + 1) % rotationKeys.size();

        float weight = SafeLinearstep(rotationKeys[firstID], rotationKeys[secondID], time);
        float4 a = rotationValues[firstID];
        float4 b = rotationValues[secondID];
        float theta = Dot44(a, b);
        a = (theta < 0.0f) ? -a : a;
        rotation = Slerp(a, b, weight);
    }

    double3 position = positionValues.back();
    if (time < positionKeys.back())
    {
        uint32_t firstID = FindCurrentIndex(positionKeys, time);
        uint32_t secondID = (firstID + 1) % positionKeys.size();

        float weight = SafeLinearstep(positionKeys[firstID], positionKeys[secondID], time);
        position = Lerp(positionValues[firstID], positionValues[secondID], double3((double)weight));
    }

    float4x4 mScale;
    mScale.SetupByScale(scale);

    float4x4 mRotation;
    mRotation.SetupByQuaternion(rotation);

    float4x4 mTranslation;
    mTranslation.SetupByTranslation( ToFloat(position) );

    mTransform = mTranslation * (mRotation * mScale);
}

void utils::NodeTree::SetAllowUpdate(utils::Scene& scene, bool parentAllowUpdate)
{
    bool allowUpdate = parentAllowUpdate || animationNodeIndex != InvalidIndex;

    for (NodeTree& child : children)
        child.SetAllowUpdate(scene, allowUpdate);

    for (uint32_t instanceIndex : instances)
    {
        Instance& instance = scene.instances[instanceIndex];
        instance.allowUpdate = allowUpdate;
    }
}

void utils::NodeTree::Animate(Scene& scene, const std::vector<AnimationNode>& animationNodes, const float4x4& parentTransform, float4x4* outTransform)
{
    const float4x4& transform = animationNodeIndex != InvalidIndex ? animationNodes[animationNodeIndex].mTransform : mTransform;
    float4x4 combinedTransform = parentTransform * transform;

    for (NodeTree& child : children)
        child.Animate(scene, animationNodes, combinedTransform, outTransform);

    if (outTransform && children.empty())
        *outTransform = combinedTransform;

    double3 position = ToDouble( combinedTransform.GetCol3().To3d() );
    combinedTransform.SetTranslation( float3::Zero() );

    for (uint32_t instanceIndex : instances)
    {
        Instance& instance = scene.instances[instanceIndex];
        instance.rotation = combinedTransform;
        instance.position = position;
    }
}

void utils::Scene::Animate(float animationSpeed, float elapsedTime, float& animationProgress, uint32_t animationIndex, float4x4* outCameraTransform)
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

    // Update animation nodes
    for (AnimationNode& animationNode : animation.animationNodes)
        animationNode.Update(t);

    // Recursively update instances in the nood tree
    animation.rootNode.Animate(*this, animation.animationNodes, mSceneToWorld);

    // Update camera animation (if requested)
    if (outCameraTransform)
    {
        float4x4 transform;
        animation.cameraNode.Animate(*this, animation.animationNodes, float4x4::Identity(), &transform);

        // Inverse 3x3 rotation (without scene-to-world rotation)
        float3 pos = transform.GetCol3().xmm;
        transform.SetTranslation( float3::Zero() );
        transform.Transpose();
        transform.SetTranslation(pos);

        // Transpose
        transform.Transpose();

        // Apply scene-to-world rotation
        float4x4 sceneToWorldT = mSceneToWorld;
        sceneToWorldT.Transpose();

        float4x4 finalTransform = mSceneToWorld * (transform * sceneToWorldT);

        // Result
        *outCameraTransform = finalTransform;
    }
}
