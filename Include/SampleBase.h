/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#define GLFW_INCLUDE_NONE 1
#include "Glfw/include/GLFW/glfw3.h"
#include "Glfw/include/GLFW/glfw3native.h"
#include "ImGui/imgui.h"

#include "NRIDescs.hpp"
#include "Extensions/NRIDeviceCreation.h"
#include "Extensions/NRISwapChain.h"
#include "Extensions/NRIHelper.h"

#include "Helper.h"
#include "Utils.h"
#include "Camera.h"

constexpr nri::SPIRVBindingOffsets SPIRV_BINDING_OFFSETS = {100, 200, 300, 400}; // see CompileHLSL.bat
constexpr bool D3D11_COMMANDBUFFER_EMULATION = false;
constexpr uint32_t DEFAULT_MEMORY_ALIGNMENT = 16;
constexpr uint32_t BUFFERED_FRAME_MAX_NUM = 2;
constexpr uint32_t SWAP_CHAIN_TEXTURE_NUM = BUFFERED_FRAME_MAX_NUM;

#undef Button4
#undef Button5

enum class Button : uint8_t
{
    Left            = GLFW_MOUSE_BUTTON_LEFT,
    Right           = GLFW_MOUSE_BUTTON_RIGHT,
    Middle          = GLFW_MOUSE_BUTTON_MIDDLE,
    Button4         = GLFW_MOUSE_BUTTON_4,
    Button5         = GLFW_MOUSE_BUTTON_5,
    Button6         = GLFW_MOUSE_BUTTON_6,
    Button7         = GLFW_MOUSE_BUTTON_7,
    Button8         = GLFW_MOUSE_BUTTON_8,

    NUM             = GLFW_MOUSE_BUTTON_LAST
};

enum class Key : uint32_t
{
    Tilda           = GLFW_KEY_GRAVE_ACCENT,
    _1              = GLFW_KEY_1,
    _2              = GLFW_KEY_2,
    _3              = GLFW_KEY_3,
    _4              = GLFW_KEY_4,
    _5              = GLFW_KEY_5,
    _6              = GLFW_KEY_6,
    _7              = GLFW_KEY_7,
    _8              = GLFW_KEY_8,
    _9              = GLFW_KEY_9,
    _0              = GLFW_KEY_0,
    Minus           = GLFW_KEY_MINUS,
    Equals          = GLFW_KEY_EQUAL,
    Back            = GLFW_KEY_BACKSPACE,

    Q               = GLFW_KEY_Q,
    W               = GLFW_KEY_W,
    E               = GLFW_KEY_E,
    R               = GLFW_KEY_R,
    T               = GLFW_KEY_T,
    Y               = GLFW_KEY_Y,
    U               = GLFW_KEY_U,
    I               = GLFW_KEY_I,
    O               = GLFW_KEY_O,
    P               = GLFW_KEY_P,
    A               = GLFW_KEY_A,
    S               = GLFW_KEY_S,
    D               = GLFW_KEY_D,
    F               = GLFW_KEY_F,
    G               = GLFW_KEY_G,
    H               = GLFW_KEY_H,
    J               = GLFW_KEY_J,
    K               = GLFW_KEY_K,
    L               = GLFW_KEY_L,
    Z               = GLFW_KEY_Z,
    X               = GLFW_KEY_X,
    C               = GLFW_KEY_C,
    V               = GLFW_KEY_V,
    B               = GLFW_KEY_B,
    N               = GLFW_KEY_N,
    M               = GLFW_KEY_M,

    Escape          = GLFW_KEY_ESCAPE,
    F1              = GLFW_KEY_F1,
    F2              = GLFW_KEY_F2,
    F3              = GLFW_KEY_F3,
    F4              = GLFW_KEY_F4,
    F5              = GLFW_KEY_F5,
    F6              = GLFW_KEY_F6,
    F7              = GLFW_KEY_F7,
    F8              = GLFW_KEY_F8,
    F9              = GLFW_KEY_F9,
    F10             = GLFW_KEY_F10,
    F11             = GLFW_KEY_F11,
    F12             = GLFW_KEY_F12,
    Pause           = GLFW_KEY_PAUSE,

    Num0            = GLFW_KEY_KP_0,
    Num1            = GLFW_KEY_KP_1,
    Num2            = GLFW_KEY_KP_2,
    Num3            = GLFW_KEY_KP_3,
    Num4            = GLFW_KEY_KP_4,
    Num5            = GLFW_KEY_KP_5,
    Num6            = GLFW_KEY_KP_6,
    Num7            = GLFW_KEY_KP_7,
    Num8            = GLFW_KEY_KP_8,
    Num9            = GLFW_KEY_KP_9,
    NumLock         = GLFW_KEY_NUM_LOCK,
    NumSub          = GLFW_KEY_KP_SUBTRACT,
    NumAdd          = GLFW_KEY_KP_ADD,
    NumDel          = GLFW_KEY_KP_DECIMAL,
    NumMul          = GLFW_KEY_KP_MULTIPLY,
    NumDiv          = GLFW_KEY_KP_DIVIDE,
    NumEnter        = GLFW_KEY_KP_ENTER,

    LControl        = GLFW_KEY_LEFT_CONTROL,
    RControl        = GLFW_KEY_RIGHT_CONTROL,
    LAlt            = GLFW_KEY_LEFT_ALT,
    RAlt            = GLFW_KEY_RIGHT_ALT,
    LShift          = GLFW_KEY_LEFT_SHIFT,
    RShift          = GLFW_KEY_RIGHT_SHIFT,
    LBracket        = GLFW_KEY_LEFT_BRACKET,
    RBracket        = GLFW_KEY_RIGHT_BRACKET,

    Scroll          = GLFW_KEY_SCROLL_LOCK,
    Tab             = GLFW_KEY_TAB,
    Return          = GLFW_KEY_ENTER,
    Semicolon       = GLFW_KEY_SEMICOLON,
    Apostrophe      = GLFW_KEY_APOSTROPHE,
    BackSlash       = GLFW_KEY_BACKSLASH,
    Comma           = GLFW_KEY_COMMA,
    Period          = GLFW_KEY_PERIOD,
    Slash           = GLFW_KEY_SLASH,
    Space           = GLFW_KEY_SPACE,
    Capital         = GLFW_KEY_CAPS_LOCK,

    Insert          = GLFW_KEY_INSERT,
    Del             = GLFW_KEY_DELETE,
    Home            = GLFW_KEY_HOME,
    End             = GLFW_KEY_END,
    PageUp          = GLFW_KEY_PAGE_UP,
    PageDown        = GLFW_KEY_PAGE_DOWN,

    Left            = GLFW_KEY_LEFT,
    Right           = GLFW_KEY_RIGHT,
    Up              = GLFW_KEY_UP,
    Down            = GLFW_KEY_DOWN,

    NUM             = GLFW_KEY_LAST
};

struct BackBuffer
{
    nri::FrameBuffer* frameBuffer;
    nri::FrameBuffer* frameBufferUI;
    nri::Descriptor* colorAttachment;
    nri::Texture* texture;
};

class SampleBase
{
public:
    SampleBase()
    {}

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

    inline uint16_t GetWindowWidth() const
    { return (uint16_t)m_WindowWidth; }

    inline uint16_t GetWindowHeight() const
    { return (uint16_t)m_WindowHeight; }

    inline uint32_t GetFrameNum() const
    { return m_FrameNum; }

    const nri::Window& GetWindow() const;
    nri::WindowSystemType GetWindowSystemType() const;

    void GetCameraDescFromInputDevices(CameraDesc& cameraDesc);
    bool CreateUserInterface(nri::Device& device, const nri::CoreInterface& coreInterface, const nri::HelperInterface& helperInterface, uint32_t windowWidth, uint32_t windowHeight, nri::Format renderTargetFormat);
    void DestroyUserInterface();
    void PrepareUserInterface();
    void RenderUserInterface(nri::CommandBuffer& commandBuffer);

    virtual bool Initialize(nri::GraphicsAPI graphicsAPI) = 0;
    virtual void PrepareFrame(uint32_t frameIndex) = 0;
    virtual void RenderFrame(uint32_t frameIndex) = 0;

    static void EnableMemoryLeakDetection(uint32_t breakOnAllocationIndex);

protected:
    Camera m_Camera;
    sFastRand m_FastRandState = {};
    uint32_t m_SwapInterval = 0;
    bool m_DebugAPI = false;
    bool m_DebugNRI = false;
    bool m_IgnoreDPI = false;
    bool m_IsActive = true;
    bool m_TestMode = false;
    std::string m_SceneFile = "ShaderBalls/ShaderBalls.obj";
    uint32_t m_DlssQuality = uint32_t(-1);

    // Private
private:
    void CursorMode(int32_t mode);

public:
    inline bool HasUserInterface() const
    { return m_timePrev != 0.0; }

    void ParseCommandLine(int32_t argc, char** argv);
    bool Create(const char* windowTitle);
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
    std::vector<nri::Memory*> m_MemoryAllocations;
    const nri::CoreInterface* NRI = nullptr;
    const nri::HelperInterface* m_Helper = nullptr;
    nri::Device* m_Device = nullptr;
    nri::DescriptorPool* m_DescriptorPool = nullptr;
    nri::DescriptorSet* m_DescriptorSet = nullptr;
    nri::Descriptor* m_FontShaderResource = nullptr;
    nri::Pipeline* m_Pipeline = nullptr;
    nri::PipelineLayout* m_PipelineLayout = nullptr;
    nri::Texture* m_FontTexture = nullptr;
    nri::Buffer* m_GeometryBuffer = nullptr;
    GLFWcursor* m_MouseCursors[ImGuiMouseCursor_COUNT] = {};
    double m_timePrev = 0.0;
    uint64_t m_StreamBufferOffset = 0;

    // Window
    GLFWwindow* m_Window = nullptr;
    nri::Window m_NRIWindow = {};
    uint32_t m_WindowWidth = 1280;
    uint32_t m_WindowHeight = 720;

    // Rendering
    uint32_t m_FrameNum = uint32_t(-1);
#if _WIN32
    nri::GraphicsAPI m_GraphicsAPI = nri::GraphicsAPI::D3D12;
#else
    nri::GraphicsAPI m_GraphicsAPI = nri::GraphicsAPI::VULKAN;
#endif
};

#define _STRINGIFY(s) #s
#define STRINGIFY(s) _STRINGIFY(s)

#define SAMPLE_MAIN(className, memoryAllocationIndexForBreak) \
    int main(int argc, char** argv) \
    { \
        SampleBase::EnableMemoryLeakDetection(memoryAllocationIndexForBreak); \
        SampleBase* sample = new className; \
        sample->ParseCommandLine(argc, argv); \
        bool result = sample->Create(STRINGIFY(PROJECT_NAME)); \
        if (result) \
            sample->RenderLoop(); \
        delete sample; \
        return result ? 0 : 1; \
    }
