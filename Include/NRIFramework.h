// © 2021 NVIDIA Corporation

#pragma once

#define NRI_FRAMEWORK_VERSION_MAJOR 0
#define NRI_FRAMEWORK_VERSION_MINOR 13
#define NRI_FRAMEWORK_VERSION_DATE "9 February 2024"
#define NRI_FRAMEWORK 1

// 3rd party
#define GLFW_INCLUDE_NONE
#include "Glfw/include/GLFW/glfw3.h"
#include "ImGui/imgui.h"

// Dependencies
#include "MathLib/MathLib.h"
#include "Timer.h"

// NRI: core & common extensions
#include "NRI.h"
#include "Extensions/NRIDeviceCreation.h"
#include "Extensions/NRISwapChain.h"
#include "Extensions/NRIHelper.h"

// NRI framework
#include "Camera.h"
#include "CmdLine.h" // https://github.com/tanakh/cmdline
#include "Controls.h"
#include "Helper.h"
#include "Utils.h"

// Settings
constexpr nri::SPIRVBindingOffsets SPIRV_BINDING_OFFSETS = {100, 200, 300, 400}; // just ShaderMake defaults for simplicity
constexpr bool D3D11_COMMANDBUFFER_EMULATION = false;
constexpr uint32_t DEFAULT_MEMORY_ALIGNMENT = 16;
constexpr uint32_t BUFFERED_FRAME_MAX_NUM = 3;
constexpr uint32_t SWAP_CHAIN_TEXTURE_NUM = 2;

struct BackBuffer
{
    nri::Descriptor* colorAttachment;
    nri::Texture* texture;
};

class SampleBase
{
public:
    SampleBase();

    virtual ~SampleBase();

    inline bool IsKeyToggled(Key key)
    {
        bool state = m_KeyToggled[(uint32_t)key];
        m_KeyToggled[(uint32_t)key] = false;

        return state;
    }

    inline bool IsKeyPressed(Key key) const
    { return m_KeyState[(uint32_t)key]; }

    inline bool IsButtonPressed(Button button) const
    { return m_ButtonState[(uint8_t)button]; }

    inline const float2& GetMouseDelta() const
    { return m_MouseDelta; }

    inline float GetMouseWheel() const
    { return m_MouseWheel; }

    inline uint2 GetWindowResolution() const
    { return m_WindowResolution; }

    inline uint2 GetOutputResolution() const
    { return m_OutputResolution; }

    inline const nri::Window& SampleBase::GetWindow() const
    { return m_NRIWindow; }

    void GetCameraDescFromInputDevices(CameraDesc& cameraDesc);
    bool CreateUserInterface(nri::Device& device, const nri::CoreInterface& coreInterface, const nri::HelperInterface& helperInterface, nri::Format renderTargetFormat);
    void DestroyUserInterface();
    void RenderUserInterface(nri::Device& device, nri::CommandBuffer& commandBuffer, float sdrScale, bool isSrgb);

    virtual void InitCmdLine([[maybe_unused]] cmdline::parser& cmdLine) { }
    virtual void ReadCmdLine([[maybe_unused]] cmdline::parser& cmdLine) { }
    virtual bool Initialize(nri::GraphicsAPI graphicsAPI) = 0;
    virtual void PrepareFrame(uint32_t frameIndex) = 0;
    virtual void RenderFrame(uint32_t frameIndex) = 0;
    virtual bool AppShouldClose() { return false; }

    static void EnableMemoryLeakDetection(uint32_t breakOnAllocationIndex);

protected:
    nri::MemoryAllocatorInterface m_MemoryAllocatorInterface = {};
    std::string m_SceneFile = "ShaderBalls/ShaderBalls.gltf";
    sFastRand m_FastRandState = {};
    GLFWwindow* m_Window = nullptr;
    Camera m_Camera;
    Timer m_Timer;
    uint2 m_OutputResolution = {1920, 1080};
    uint2 m_WindowResolution = {};
    uint8_t m_VsyncInterval = 0;
    uint32_t m_DpiMode = 0;
    float m_MouseSensitivity = 1.0f;
    bool m_DebugAPI = false;
    bool m_DebugNRI = false;
    bool m_IsActive = true;

    // Private
private:
    void CursorMode(int32_t mode);
    void PrepareUserInterface();

public:
    inline bool HasUserInterface() const
    { return m_timePrev != 0.0; }

    void InitCmdLineDefault(cmdline::parser& cmdLine);
    void ReadCmdLineDefault(cmdline::parser& cmdLine);
    bool Create(int32_t argc, char** argv, const char* windowTitle);
    void RenderLoop();

    // Input (not public)
public:
    std::array<bool, (size_t)Key::NUM> m_KeyState = {};
    std::array<bool, (size_t)Key::NUM> m_KeyToggled = {};
    std::array<bool, (size_t)Button::NUM> m_ButtonState = {};
    std::array<bool, (size_t)Button::NUM> m_ButtonJustPressed = {};
    float2 m_MouseDelta = {};
    float2 m_MousePosPrev = {};
    float m_MouseWheel = 0.0f;

private:
    // UI
    const nri::CoreInterface* NRI = nullptr;
    const nri::HelperInterface* m_Helper = nullptr;
    nri::Device* m_Device = nullptr;
    nri::DescriptorPool* m_DescriptorPool = nullptr;
    nri::DescriptorSet* m_DescriptorSet = nullptr;
    nri::Descriptor* m_FontShaderResource = nullptr;
    nri::Descriptor* m_Sampler = nullptr;
    nri::Pipeline* m_Pipeline = nullptr;
    nri::PipelineLayout* m_PipelineLayout = nullptr;
    nri::Texture* m_FontTexture = nullptr;
    nri::Memory* m_FontTextureMemory = nullptr;
    nri::Buffer* m_GeometryBuffer = nullptr;
    nri::Memory* m_GeometryBufferMemory = nullptr;
    GLFWcursor* m_MouseCursors[ImGuiMouseCursor_COUNT] = {};
    double m_timePrev = 0.0;
    uint64_t m_StreamBufferOffset = 0;
    nri::Window m_NRIWindow = {};

    // Rendering
    uint32_t m_FrameNum = uint32_t(-1);
    uint32_t m_StreamBufferSize = 0;
};

#define _STRINGIFY(s) #s
#define STRINGIFY(s) _STRINGIFY(s)

#define SAMPLE_MAIN(className, memoryAllocationIndexForBreak) \
    int main(int argc, char** argv) \
    { \
        SampleBase::EnableMemoryLeakDetection(memoryAllocationIndexForBreak); \
        SampleBase* sample = new className; \
        bool result = sample->Create(argc, argv, STRINGIFY(PROJECT_NAME)); \
        if (result) \
            sample->RenderLoop(); \
        delete sample; \
        return result ? 0 : 1; \
    }
