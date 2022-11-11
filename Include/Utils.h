/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <string>
#include <vector>

#undef OPAQUE
#undef TRANSPARENT

namespace utils
{
    struct Texture;
    struct Scene;
    typedef std::vector<std::vector<uint8_t>> ShaderCodeStorage;
    typedef void* Mip;

    enum StaticTexture : uint32_t
    {
        Invalid,
        Black,
        FlatNormal,
        ScramblingRanking1spp,
        SobolSequence
    };

    enum class AlphaMode
    {
        OPAQUE,
        PREMULTIPLIED,
        TRANSPARENT,
        OFF // alpha is 0 everywhere
    };

    enum class DataFolder
    {
        ROOT,
        SHADERS,
        TEXTURES,
        SCENES,
        TESTS
    };

    const char* GetFileName(const std::string& path);
    std::string GetFullPath(const std::string& localPath, DataFolder dataFolder);
    bool LoadFile(const std::string& path, std::vector<uint8_t>& data);
    nri::ShaderDesc LoadShader(nri::GraphicsAPI graphicsAPI, const std::string& path, ShaderCodeStorage& storage, const char* entryPointName = nullptr);
    bool LoadTexture(const std::string& path, Texture& texture, bool computeAvgColorAndAlphaMode = false);
    void LoadTextureFromMemory(nri::Format format, uint32_t width, uint32_t height, const uint8_t *pixels, Texture &texture);
    bool LoadScene(const std::string& path, Scene& scene, bool simpleOIT = false, const std::vector<float3>& instanceData = {float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f) });

    struct Texture
    {
        Mip* mips = nullptr;
        std::string name;
        float4 avgColor = float4::Zero();
        uint64_t hash = 0;
        AlphaMode alphaMode = AlphaMode::OPAQUE;
        nri::Format format = nri::Format::UNKNOWN;
        uint16_t width = 0;
        uint16_t height = 0;
        uint16_t depth = 0;
        uint16_t mipNum = 0;
        uint16_t arraySize = 0;

        ~Texture();

        bool IsBlockCompressed() const;
        void GetSubresource(nri::TextureSubresourceUploadDesc& subresource, uint32_t mipIndex, uint32_t arrayIndex = 0) const;

        inline void OverrideFormat(nri::Format fmt)
        { this->format = fmt; }

        inline uint16_t GetArraySize() const
        { return arraySize; }

        inline uint16_t GetMipNum() const
        { return mipNum; }

        inline uint16_t GetWidth() const
        { return width; }

        inline uint16_t GetHeight() const
        { return height; }

        inline uint16_t GetDepth() const
        { return depth; }

        inline nri::Format GetFormat() const
        { return format; }
    };

    struct MaterialGroup
    {
        uint32_t materialOffset;
        uint32_t materialNum;
    };

    struct Material
    {
        float4 avgBaseColor;
        float4 avgSpecularColor;
        uint32_t instanceOffset;
        uint32_t instanceNum;
        uint32_t diffuseMapIndex;
        uint32_t specularMapIndex;
        uint32_t normalMapIndex;
        uint32_t emissiveMapIndex;
        AlphaMode alphaMode;
        bool isEmissive;

        inline bool IsOpaque() const
        { return alphaMode == AlphaMode::OPAQUE; }

        inline bool IsAlphaOpaque() const
        { return alphaMode == AlphaMode::PREMULTIPLIED; }

        inline bool IsTransparent() const
        { return alphaMode == AlphaMode::TRANSPARENT; }

        inline bool IsOff() const
        { return alphaMode == AlphaMode::OFF; }

        inline bool IsEmissive() const
        { return isEmissive; }
    };

    struct Instance
    {
        float4x4 rotation;
        float4x4 rotationPrev;
        double3 position;
        double3 positionPrev;
        uint32_t meshIndex;
        uint32_t materialIndex;
    };

    struct Mesh
    {
        cBoxf aabb;
        uint32_t vertexOffset;
        uint32_t indexOffset;
        uint32_t indexNum;
        uint32_t vertexNum;
    };

    struct Vertex
    {
        float position[3];
        uint32_t uv; // half float
        uint32_t normal; // 10 10 10 2 unorm
        uint32_t tangent; // 10 10 10 2 unorm (.w - handedness)
    };

    struct UnpackedVertex
    {
        float position[3];
        float uv[2];
        float normal[3];
        float tangent[4];
    };

    struct Primitive
    {
        float worldToUvUnits;
        float curvature;
    };

    struct AnimationNode
    {
        std::vector<double3> positionValues;
        std::vector<float4> rotationValues;
        std::vector<float3> scaleValues;
        std::vector<float> positionKeys;
        std::vector<float> rotationKeys;
        std::vector<float> scaleKeys;
        float4x4 mTransform = float4x4::Identity();

        void Animate(float time);
    };

    struct NodeTree
    {
        std::vector<NodeTree> children;
        std::vector<uint32_t> instances;
        float4x4 mTransform = float4x4::Identity();
        uint64_t hash = 0;
        uint32_t animationNode = uint32_t(-1);

        inline bool HasAnimation() const
        { return animationNode != uint32_t(-1); }

        void Animate(utils::Scene& scene, std::vector<AnimationNode>& animationNodes, const float4x4& parentTransform, float4x4* outTransform = nullptr);
    };

    struct Animation
    {
        std::vector<AnimationNode> animationNodes;
        NodeTree rootNode;
        NodeTree cameraNode;
        std::string animationName;
        float durationMs = 0.0f;
        float animationProgress;
        float sign = 1.0f;
        float normalizedTime;
        bool hasCameraAnimation;
    };

    typedef uint16_t Index;

    struct Scene
    {
        ~Scene()
        {
            UnloadTextureData();
        }

        // Transient resources - texture & geometry data (can be unloaded after uploading on GPU)
        std::vector<utils::Texture*> textures;
        std::vector<Vertex> vertices;
        std::vector<UnpackedVertex> unpackedVertices;
        std::vector<Index> indices;
        std::vector<Primitive> primitives;

        // Other resources
        std::vector<MaterialGroup> materialsGroups; // 0 - opaque, 1 - two-sided, alpha opaque, 2 - transparent (back faces), 3 - transparent (front faces)
        std::vector<Material> materials;
        std::vector<Instance> instances;
        std::vector<Mesh> meshes;
        std::vector<Animation> animations;
        float4x4 mSceneToWorld = float4x4::Identity();
        cBoxf aabb;

        void Animate(float animationSpeed, float elapsedTime, float& animationProgress, uint32_t animationID, float4x4* outCameraTransform = nullptr);

        inline void UnloadTextureData()
        {
            for (auto texture : textures)
                delete texture;

            textures.resize(0);
            textures.shrink_to_fit();
        }

        inline void UnloadGeometryData()
        {
            vertices.resize(0);
            vertices.shrink_to_fit();

            unpackedVertices.resize(0);
            unpackedVertices.shrink_to_fit();

            indices.resize(0);
            indices.shrink_to_fit();

            primitives.resize(0);
            primitives.shrink_to_fit();
        }
    };
}

#define NRI_ABORT_ON_FAILURE(result) \
    if ((result) != nri::Result::SUCCESS) \
        exit(1);

#define NRI_ABORT_ON_FALSE(result) \
    if (!(result)) \
        exit(1);

