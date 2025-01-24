// © 2021 NVIDIA Corporation

#include "NRIFramework.h"

#ifdef NRI_ENABLE_AGILITY_SDK_SUPPORT
    #include "NRIAgilitySDK.h"
#endif

#if defined _WIN32
#    define GLFW_EXPOSE_NATIVE_WIN32
#elif defined __linux__
#    define GLFW_EXPOSE_NATIVE_X11
#elif defined __APPLE__
#    define GLFW_EXPOSE_NATIVE_COCOA
#    include "MetalUtility/MetalUtility.h"
#else
#    error "Unknown platform"
#endif
#include "Glfw/include/GLFW/glfw3native.h"

#if defined(__linux__) || defined(__APPLE__)
#    include <csignal>
#endif

#include <thread>

template <typename T>
constexpr void MaybeUnused([[maybe_unused]] const T& arg) {
}

void CreateDebugAllocator(nri::AllocationCallbacks& allocationCallbacks);
void DestroyDebugAllocator(nri::AllocationCallbacks& allocationCallbacks);

//==================================================================================================================================================
// MEMORY
//==================================================================================================================================================

#if _WIN32

void* __CRTDECL operator new(size_t size) {
    return _aligned_malloc(size, DEFAULT_MEMORY_ALIGNMENT);
}

void* __CRTDECL operator new[](size_t size) {
    return _aligned_malloc(size, DEFAULT_MEMORY_ALIGNMENT);
}

void __CRTDECL operator delete(void* p) noexcept {
    _aligned_free(p);
}

void __CRTDECL operator delete[](void* p) noexcept {
    _aligned_free(p);
}

#endif

//==================================================================================================================================================
// GLFW CALLBACKS
//==================================================================================================================================================

static ImGuiKey RemapKey(int32_t key) {
    switch (key) {
        case GLFW_KEY_SPACE:
            return ImGuiKey_Space;
        case GLFW_KEY_APOSTROPHE:
            return ImGuiKey_Apostrophe;
        case GLFW_KEY_COMMA:
            return ImGuiKey_Comma;
        case GLFW_KEY_MINUS:
            return ImGuiKey_Minus;
        case GLFW_KEY_PERIOD:
            return ImGuiKey_Period;
        case GLFW_KEY_SLASH:
            return ImGuiKey_Slash;
        case GLFW_KEY_0:
            return ImGuiKey_0;
        case GLFW_KEY_1:
            return ImGuiKey_1;
        case GLFW_KEY_2:
            return ImGuiKey_2;
        case GLFW_KEY_3:
            return ImGuiKey_3;
        case GLFW_KEY_4:
            return ImGuiKey_4;
        case GLFW_KEY_5:
            return ImGuiKey_5;
        case GLFW_KEY_6:
            return ImGuiKey_6;
        case GLFW_KEY_7:
            return ImGuiKey_7;
        case GLFW_KEY_8:
            return ImGuiKey_8;
        case GLFW_KEY_9:
            return ImGuiKey_9;
        case GLFW_KEY_SEMICOLON:
            return ImGuiKey_Semicolon;
        case GLFW_KEY_EQUAL:
            return ImGuiKey_Equal;
        case GLFW_KEY_A:
            return ImGuiKey_A;
        case GLFW_KEY_B:
            return ImGuiKey_B;
        case GLFW_KEY_C:
            return ImGuiKey_C;
        case GLFW_KEY_D:
            return ImGuiKey_D;
        case GLFW_KEY_E:
            return ImGuiKey_E;
        case GLFW_KEY_F:
            return ImGuiKey_F;
        case GLFW_KEY_G:
            return ImGuiKey_G;
        case GLFW_KEY_H:
            return ImGuiKey_H;
        case GLFW_KEY_I:
            return ImGuiKey_I;
        case GLFW_KEY_J:
            return ImGuiKey_J;
        case GLFW_KEY_K:
            return ImGuiKey_K;
        case GLFW_KEY_L:
            return ImGuiKey_L;
        case GLFW_KEY_M:
            return ImGuiKey_M;
        case GLFW_KEY_N:
            return ImGuiKey_N;
        case GLFW_KEY_O:
            return ImGuiKey_O;
        case GLFW_KEY_P:
            return ImGuiKey_P;
        case GLFW_KEY_Q:
            return ImGuiKey_Q;
        case GLFW_KEY_R:
            return ImGuiKey_R;
        case GLFW_KEY_S:
            return ImGuiKey_S;
        case GLFW_KEY_T:
            return ImGuiKey_T;
        case GLFW_KEY_U:
            return ImGuiKey_U;
        case GLFW_KEY_V:
            return ImGuiKey_V;
        case GLFW_KEY_W:
            return ImGuiKey_W;
        case GLFW_KEY_X:
            return ImGuiKey_X;
        case GLFW_KEY_Y:
            return ImGuiKey_Y;
        case GLFW_KEY_Z:
            return ImGuiKey_Z;
        case GLFW_KEY_LEFT_BRACKET:
            return ImGuiKey_LeftBracket;
        case GLFW_KEY_BACKSLASH:
            return ImGuiKey_Backslash;
        case GLFW_KEY_RIGHT_BRACKET:
            return ImGuiKey_RightBracket;
        case GLFW_KEY_GRAVE_ACCENT:
            return ImGuiKey_GraveAccent;
        case GLFW_KEY_ESCAPE:
            return ImGuiKey_Escape;
        case GLFW_KEY_ENTER:
            return ImGuiKey_Enter;
        case GLFW_KEY_TAB:
            return ImGuiKey_Tab;
        case GLFW_KEY_BACKSPACE:
            return ImGuiKey_Backspace;
        case GLFW_KEY_INSERT:
            return ImGuiKey_Insert;
        case GLFW_KEY_DELETE:
            return ImGuiKey_Delete;
        case GLFW_KEY_RIGHT:
            return ImGuiKey_RightArrow;
        case GLFW_KEY_LEFT:
            return ImGuiKey_LeftArrow;
        case GLFW_KEY_DOWN:
            return ImGuiKey_DownArrow;
        case GLFW_KEY_UP:
            return ImGuiKey_UpArrow;
        case GLFW_KEY_PAGE_UP:
            return ImGuiKey_PageUp;
        case GLFW_KEY_PAGE_DOWN:
            return ImGuiKey_PageDown;
        case GLFW_KEY_HOME:
            return ImGuiKey_Home;
        case GLFW_KEY_END:
            return ImGuiKey_End;
        case GLFW_KEY_CAPS_LOCK:
            return ImGuiKey_CapsLock;
        case GLFW_KEY_SCROLL_LOCK:
            return ImGuiKey_ScrollLock;
        case GLFW_KEY_NUM_LOCK:
            return ImGuiKey_NumLock;
        case GLFW_KEY_PRINT_SCREEN:
            return ImGuiKey_PrintScreen;
        case GLFW_KEY_PAUSE:
            return ImGuiKey_Pause;
        case GLFW_KEY_F1:
            return ImGuiKey_F1;
        case GLFW_KEY_F2:
            return ImGuiKey_F2;
        case GLFW_KEY_F3:
            return ImGuiKey_F3;
        case GLFW_KEY_F4:
            return ImGuiKey_F4;
        case GLFW_KEY_F5:
            return ImGuiKey_F5;
        case GLFW_KEY_F6:
            return ImGuiKey_F6;
        case GLFW_KEY_F7:
            return ImGuiKey_F7;
        case GLFW_KEY_F8:
            return ImGuiKey_F8;
        case GLFW_KEY_F9:
            return ImGuiKey_F9;
        case GLFW_KEY_F10:
            return ImGuiKey_F10;
        case GLFW_KEY_F11:
            return ImGuiKey_F11;
        case GLFW_KEY_F12:
            return ImGuiKey_F12;
        case GLFW_KEY_KP_0:
            return ImGuiKey_Keypad0;
        case GLFW_KEY_KP_1:
            return ImGuiKey_Keypad1;
        case GLFW_KEY_KP_2:
            return ImGuiKey_Keypad2;
        case GLFW_KEY_KP_3:
            return ImGuiKey_Keypad3;
        case GLFW_KEY_KP_4:
            return ImGuiKey_Keypad4;
        case GLFW_KEY_KP_5:
            return ImGuiKey_Keypad5;
        case GLFW_KEY_KP_6:
            return ImGuiKey_Keypad6;
        case GLFW_KEY_KP_7:
            return ImGuiKey_Keypad7;
        case GLFW_KEY_KP_8:
            return ImGuiKey_Keypad8;
        case GLFW_KEY_KP_9:
            return ImGuiKey_Keypad9;
        case GLFW_KEY_KP_DECIMAL:
            return ImGuiKey_KeypadDecimal;
        case GLFW_KEY_KP_DIVIDE:
            return ImGuiKey_KeypadDivide;
        case GLFW_KEY_KP_MULTIPLY:
            return ImGuiKey_KeypadMultiply;
        case GLFW_KEY_KP_SUBTRACT:
            return ImGuiKey_KeypadSubtract;
        case GLFW_KEY_KP_ADD:
            return ImGuiKey_KeypadAdd;
        case GLFW_KEY_KP_ENTER:
            return ImGuiKey_KeypadEnter;
        case GLFW_KEY_KP_EQUAL:
            return ImGuiKey_KeypadEqual;
        case GLFW_KEY_LEFT_SHIFT:
            return ImGuiKey_LeftShift;
        case GLFW_KEY_LEFT_CONTROL:
            return ImGuiKey_LeftCtrl;
        case GLFW_KEY_LEFT_ALT:
            return ImGuiKey_LeftAlt;
        case GLFW_KEY_LEFT_SUPER:
            return ImGuiKey_LeftSuper;
        case GLFW_KEY_RIGHT_SHIFT:
            return ImGuiKey_RightShift;
        case GLFW_KEY_RIGHT_CONTROL:
            return ImGuiKey_RightCtrl;
        case GLFW_KEY_RIGHT_ALT:
            return ImGuiKey_RightAlt;
        case GLFW_KEY_RIGHT_SUPER:
            return ImGuiKey_RightSuper;
        case GLFW_KEY_MENU:
            return ImGuiKey_Menu;
        default:
            return ImGuiKey_None;
    }
}

static void GLFW_ErrorCallback(int32_t error, const char* message) {
    printf("GLFW error[%d]: %s\n", error, message);
#if _WIN32
    DebugBreak();
#else
    raise(SIGTRAP);
#endif
}

static void GLFW_KeyCallback(GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods) {
    MaybeUnused(scancode);
    MaybeUnused(mods);

    SampleBase* p = (SampleBase*)glfwGetWindowUserPointer(window);

    if (key < 0)
        return;

    p->m_KeyState[key] = action != GLFW_RELEASE;
    if (action != GLFW_RELEASE)
        p->m_KeyToggled[key] = true;

    if (p->HasUserInterface()) {
        ImGuiIO& io = ImGui::GetIO();
        ImGuiKey remappedKey = RemapKey(key);
        io.AddKeyEvent(remappedKey, action == GLFW_PRESS);
    }
}

static void GLFW_CharCallback(GLFWwindow* window, uint32_t codepoint) {
    SampleBase* p = (SampleBase*)glfwGetWindowUserPointer(window);

    if (p->HasUserInterface()) {
        ImGuiIO& io = ImGui::GetIO();
        io.AddInputCharacter(codepoint);
    }
}

static void GLFW_ButtonCallback(GLFWwindow* window, int32_t button, int32_t action, int32_t mods) {
    MaybeUnused(mods);

    SampleBase* p = (SampleBase*)glfwGetWindowUserPointer(window);

    p->m_ButtonState[button] = action != GLFW_RELEASE;
    p->m_ButtonJustPressed[button] = action != GLFW_RELEASE;
}

static void GLFW_CursorPosCallback(GLFWwindow* window, double x, double y) {
    SampleBase* p = (SampleBase*)glfwGetWindowUserPointer(window);

    float2 curPos = float2(float(x), float(y));
    p->m_MouseDelta = curPos - p->m_MousePosPrev;
}

static void GLFW_ScrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    SampleBase* p = (SampleBase*)glfwGetWindowUserPointer(window);

    p->m_MouseWheel = (float)yoffset;

    if (p->HasUserInterface()) {
        ImGuiIO& io = ImGui::GetIO();
        io.MouseWheelH += (float)xoffset;
        io.MouseWheel += (float)yoffset;
    }
}

//==================================================================================================================================================
// SAMPLE BASE
//==================================================================================================================================================

SampleBase::SampleBase() {
#if _DEBUG
    CreateDebugAllocator(m_AllocationCallbacks);
#endif
}

SampleBase::~SampleBase() {
    glfwTerminate();

#if _DEBUG
    if (m_AllocationCallbacks.userArg != nullptr)
        DestroyDebugAllocator(m_AllocationCallbacks);
#endif
}

void SampleBase::GetCameraDescFromInputDevices(CameraDesc& cameraDesc) {
    cameraDesc.timeScale = 0.025f * m_Timer.GetSmoothedFrameTime();

    if (!IsButtonPressed(Button::Right)) {
        CursorMode(GLFW_CURSOR_NORMAL);
        return;
    }

    CursorMode(GLFW_CURSOR_DISABLED);

    if (GetMouseWheel() > 0.0f)
        m_Camera.state.motionScale *= 1.1f;
    else if (GetMouseWheel() < 0.0f)
        m_Camera.state.motionScale /= 1.1f;

    float motionScale = m_Camera.state.motionScale;

    float2 mouseDelta = GetMouseDelta();
    cameraDesc.dYaw = -mouseDelta.x * m_MouseSensitivity;
    cameraDesc.dPitch = -mouseDelta.y * m_MouseSensitivity;

    if (IsKeyPressed(Key::Right))
        cameraDesc.dYaw -= motionScale;
    if (IsKeyPressed(Key::Left))
        cameraDesc.dYaw += motionScale;

    if (IsKeyPressed(Key::Up))
        cameraDesc.dPitch += motionScale;
    if (IsKeyPressed(Key::Down))
        cameraDesc.dPitch -= motionScale;

    if (IsKeyPressed(Key::W))
        cameraDesc.dLocal.z += motionScale;
    if (IsKeyPressed(Key::S))
        cameraDesc.dLocal.z -= motionScale;
    if (IsKeyPressed(Key::D))
        cameraDesc.dLocal.x += motionScale;
    if (IsKeyPressed(Key::A))
        cameraDesc.dLocal.x -= motionScale;
    if (IsKeyPressed(Key::E))
        cameraDesc.dLocal.y += motionScale;
    if (IsKeyPressed(Key::Q))
        cameraDesc.dLocal.y -= motionScale;
}

struct ImDrawVertOpt {
    float pos[2];
    uint32_t uv;
    uint32_t col;
};

bool SampleBase::InitUI(const nri::CoreInterface& NRI, const nri::HelperInterface& helperInterface, nri::Device& device, nri::Format renderTargetFormat) {
    // ImGui setup
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    float contentScale = 1.0f;
    if (m_DpiMode != 0) {
        GLFWmonitor* monitor = glfwGetPrimaryMonitor();

        float unused;
        glfwGetMonitorContentScale(monitor, &contentScale, &unused);
    }

    ImGuiStyle& style = ImGui::GetStyle();
    style.FrameBorderSize = 1;
    style.WindowBorderSize = 1;
    style.ScaleAllSizes(contentScale);

    ImGuiIO& io = ImGui::GetIO();
    io.BackendFlags |= ImGuiBackendFlags_HasMouseCursors; // We can honor GetMouseCursor() values (optional)
    io.BackendFlags |= ImGuiBackendFlags_HasSetMousePos;  // We can honor io.WantSetMousePos requests (optional, rarely used)
    io.IniFilename = nullptr;

    m_MouseCursors[ImGuiMouseCursor_Arrow] = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
    m_MouseCursors[ImGuiMouseCursor_TextInput] = glfwCreateStandardCursor(GLFW_IBEAM_CURSOR);
    m_MouseCursors[ImGuiMouseCursor_ResizeAll] = glfwCreateStandardCursor(GLFW_ARROW_CURSOR); // FIXME: GLFW doesn't have this.
    m_MouseCursors[ImGuiMouseCursor_ResizeNS] = glfwCreateStandardCursor(GLFW_VRESIZE_CURSOR);
    m_MouseCursors[ImGuiMouseCursor_ResizeEW] = glfwCreateStandardCursor(GLFW_HRESIZE_CURSOR);
    m_MouseCursors[ImGuiMouseCursor_ResizeNESW] = glfwCreateStandardCursor(GLFW_ARROW_CURSOR); // FIXME: GLFW doesn't have this.
    m_MouseCursors[ImGuiMouseCursor_ResizeNWSE] = glfwCreateStandardCursor(GLFW_ARROW_CURSOR); // FIXME: GLFW doesn't have this.
    m_MouseCursors[ImGuiMouseCursor_Hand] = glfwCreateStandardCursor(GLFW_HAND_CURSOR);

    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(device);

    // Pipeline
    {
        nri::DescriptorRangeDesc descriptorRanges[] = {
            {0, 1, nri::DescriptorType::TEXTURE, nri::StageBits::FRAGMENT_SHADER},
            {0, 1, nri::DescriptorType::SAMPLER, nri::StageBits::FRAGMENT_SHADER},
        };

        nri::DescriptorSetDesc descriptorSet = {0, descriptorRanges, helper::GetCountOf(descriptorRanges)};

        nri::RootConstantDesc rootConstants = {};
        rootConstants.registerIndex = 0;
        rootConstants.size = 16;
        rootConstants.shaderStages = nri::StageBits::ALL;

        nri::PipelineLayoutDesc pipelineLayoutDesc = {};
        pipelineLayoutDesc.descriptorSetNum = 1;
        pipelineLayoutDesc.descriptorSets = &descriptorSet;
        pipelineLayoutDesc.rootConstantNum = 1;
        pipelineLayoutDesc.rootConstants = &rootConstants;
        pipelineLayoutDesc.shaderStages = nri::StageBits::VERTEX_SHADER | nri::StageBits::FRAGMENT_SHADER;

        if (NRI.CreatePipelineLayout(device, pipelineLayoutDesc, m_PipelineLayout) != nri::Result::SUCCESS)
            return false;

        utils::ShaderCodeStorage shaderCodeStorage;
        nri::ShaderDesc shaders[] = {
            utils::LoadShader(deviceDesc.graphicsAPI, "UI.vs", shaderCodeStorage),
            utils::LoadShader(deviceDesc.graphicsAPI, "UI.fs", shaderCodeStorage),
        };

        nri::VertexStreamDesc vertexStreamDesc = {};
        vertexStreamDesc.bindingSlot = 0;
        vertexStreamDesc.stride = sizeof(ImDrawVertOpt);

        nri::VertexAttributeDesc vertexAttributeDesc[3] = {};
        {
            vertexAttributeDesc[0].format = nri::Format::RG32_SFLOAT;
            vertexAttributeDesc[0].streamIndex = 0;
            vertexAttributeDesc[0].offset = helper::GetOffsetOf(&ImDrawVertOpt::pos);
            vertexAttributeDesc[0].d3d = {"POSITION", 0};
            vertexAttributeDesc[0].vk = {0};

            vertexAttributeDesc[1].format = nri::Format::RG16_UNORM;
            vertexAttributeDesc[1].streamIndex = 0;
            vertexAttributeDesc[1].offset = helper::GetOffsetOf(&ImDrawVertOpt::uv);
            vertexAttributeDesc[1].d3d = {"TEXCOORD", 0};
            vertexAttributeDesc[1].vk = {1};

            vertexAttributeDesc[2].format = nri::Format::RGBA8_UNORM;
            vertexAttributeDesc[2].streamIndex = 0;
            vertexAttributeDesc[2].offset = helper::GetOffsetOf(&ImDrawVertOpt::col);
            vertexAttributeDesc[2].d3d = {"COLOR", 0};
            vertexAttributeDesc[2].vk = {2};
        }

        nri::VertexInputDesc vertexInputDesc = {};
        vertexInputDesc.attributes = vertexAttributeDesc;
        vertexInputDesc.attributeNum = (uint8_t)helper::GetCountOf(vertexAttributeDesc);
        vertexInputDesc.streams = &vertexStreamDesc;
        vertexInputDesc.streamNum = 1;

        nri::InputAssemblyDesc inputAssemblyDesc = {};
        inputAssemblyDesc.topology = nri::Topology::TRIANGLE_LIST;

        nri::RasterizationDesc rasterizationDesc = {};
        rasterizationDesc.fillMode = nri::FillMode::SOLID;
        rasterizationDesc.cullMode = nri::CullMode::NONE;

        nri::ColorAttachmentDesc colorAttachmentDesc = {};
        colorAttachmentDesc.format = renderTargetFormat;
        colorAttachmentDesc.colorWriteMask = nri::ColorWriteBits::RGBA;
        colorAttachmentDesc.blendEnabled = true;
        colorAttachmentDesc.colorBlend = {nri::BlendFactor::SRC_ALPHA, nri::BlendFactor::ONE_MINUS_SRC_ALPHA, nri::BlendFunc::ADD};
        colorAttachmentDesc.alphaBlend = {nri::BlendFactor::ONE_MINUS_SRC_ALPHA, nri::BlendFactor::ZERO, nri::BlendFunc::ADD};

        nri::OutputMergerDesc outputMergerDesc = {};
        outputMergerDesc.colors = &colorAttachmentDesc;
        outputMergerDesc.colorNum = 1;

        nri::GraphicsPipelineDesc graphicsPipelineDesc = {};
        graphicsPipelineDesc.pipelineLayout = m_PipelineLayout;
        graphicsPipelineDesc.vertexInput = &vertexInputDesc;
        graphicsPipelineDesc.inputAssembly = inputAssemblyDesc;
        graphicsPipelineDesc.rasterization = rasterizationDesc;
        graphicsPipelineDesc.outputMerger = outputMergerDesc;
        graphicsPipelineDesc.shaders = shaders;
        graphicsPipelineDesc.shaderNum = helper::GetCountOf(shaders);

        if (NRI.CreateGraphicsPipeline(device, graphicsPipelineDesc, m_Pipeline) != nri::Result::SUCCESS)
            return false;
    }

    ImFontConfig fontConfig = {};
    fontConfig.SizePixels = floor(13.0f * contentScale);
    io.Fonts->AddFontDefault(&fontConfig);

    int32_t fontWidth = 0, fontHeight = 0;
    uint8_t* fontPixels = nullptr;
    io.Fonts->GetTexDataAsAlpha8(&fontPixels, &fontWidth, &fontHeight);

    // Texture
    constexpr nri::Format format = nri::Format::R8_UNORM;

    nri::TextureDesc textureDesc = {};
    textureDesc.type = nri::TextureType::TEXTURE_2D;
    textureDesc.format = format;
    textureDesc.width = (uint16_t)fontWidth;
    textureDesc.height = (uint16_t)fontHeight;
    textureDesc.mipNum = 1;
    textureDesc.usage = nri::TextureUsageBits::SHADER_RESOURCE;

    if (NRI.CreateTexture(device, textureDesc, m_FontTexture) != nri::Result::SUCCESS)
        return false;

    nri::ResourceGroupDesc resourceGroupDesc = {};
    resourceGroupDesc.memoryLocation = nri::MemoryLocation::DEVICE;
    resourceGroupDesc.textureNum = 1;
    resourceGroupDesc.textures = &m_FontTexture;

    nri::Result result = helperInterface.AllocateAndBindMemory(device, resourceGroupDesc, &m_FontTextureMemory);
    if (result != nri::Result::SUCCESS)
        return false;

    // Descriptor - texture
    nri::Texture2DViewDesc texture2DViewDesc = {m_FontTexture, nri::Texture2DViewType::SHADER_RESOURCE_2D, format};
    if (NRI.CreateTexture2DView(texture2DViewDesc, m_FontShaderResource) != nri::Result::SUCCESS)
        return false;

    utils::Texture texture;
    utils::LoadTextureFromMemory(format, fontWidth, fontHeight, fontPixels, texture);

    // Descriptor - sampler
    nri::SamplerDesc samplerDesc = {};
    samplerDesc.anisotropy = 1;
    samplerDesc.addressModes = {nri::AddressMode::REPEAT, nri::AddressMode::REPEAT};
    samplerDesc.filters.min = contentScale > 1.25f ? nri::Filter::NEAREST : nri::Filter::LINEAR;
    samplerDesc.filters.mag = contentScale > 1.25f ? nri::Filter::NEAREST : nri::Filter::LINEAR;
    samplerDesc.filters.mip = contentScale > 1.25f ? nri::Filter::NEAREST : nri::Filter::LINEAR;

    if (NRI.CreateSampler(device, samplerDesc, m_Sampler) != nri::Result::SUCCESS)
        return false;

    // Upload data
    nri::Queue* queue = nullptr;
    NRI.GetQueue(device, nri::QueueType::GRAPHICS, 0, queue);
    {
        nri::TextureSubresourceUploadDesc subresource = {};
        texture.GetSubresource(subresource, 0);

        nri::TextureUploadDesc textureData = {};
        textureData.subresources = &subresource;
        textureData.texture = m_FontTexture;
        textureData.after = {nri::AccessBits::SHADER_RESOURCE, nri::Layout::SHADER_RESOURCE};

        if (helperInterface.UploadData(*queue, &textureData, 1, nullptr, 0) != nri::Result::SUCCESS)
            return false;
    }

    // Descriptor pool
    {
        nri::DescriptorPoolDesc descriptorPoolDesc = {};
        descriptorPoolDesc.descriptorSetMaxNum = 1;
        descriptorPoolDesc.textureMaxNum = 1;
        descriptorPoolDesc.samplerMaxNum = 1;

        if (NRI.CreateDescriptorPool(device, descriptorPoolDesc, m_DescriptorPool) != nri::Result::SUCCESS)
            return false;
    }

    // Descriptor set
    {
        if (NRI.AllocateDescriptorSets(*m_DescriptorPool, *m_PipelineLayout, 0, &m_DescriptorSet, 1, 0) != nri::Result::SUCCESS)
            return false;

        nri::DescriptorRangeUpdateDesc descriptorRangeUpdateDesc[] = {
            {&m_FontShaderResource, 1},
            {&m_Sampler, 1}};

        NRI.UpdateDescriptorRanges(*m_DescriptorSet, 0, helper::GetCountOf(descriptorRangeUpdateDesc), descriptorRangeUpdateDesc);
    }

    m_TimePrev = glfwGetTime();

    return true;
}

void SampleBase::DestroyUI(const nri::CoreInterface& NRI) {
    if (!HasUserInterface())
        return;

    ImGui::DestroyContext();

    NRI.DestroyDescriptorPool(*m_DescriptorPool);
    NRI.DestroyPipeline(*m_Pipeline);
    NRI.DestroyPipelineLayout(*m_PipelineLayout);
    NRI.DestroyDescriptor(*m_Sampler);
    NRI.DestroyDescriptor(*m_FontShaderResource);
    NRI.DestroyTexture(*m_FontTexture);
    NRI.FreeMemory(*m_FontTextureMemory);
}

void SampleBase::BeginUI() {
    if (!HasUserInterface())
        return;

    ImGuiIO& io = ImGui::GetIO();

    // Setup time step
    double timeCur = glfwGetTime();
    io.DeltaTime = (float)(timeCur - m_TimePrev);
    io.DisplaySize = ImVec2((float)m_WindowResolution.x, (float)m_WindowResolution.y);
    m_TimePrev = timeCur;

    // Update key modifiers
    io.AddKeyEvent(ImGuiMod_Ctrl, IsKeyPressed(Key::LControl) || IsKeyPressed(Key::RControl));
    io.AddKeyEvent(ImGuiMod_Shift, IsKeyPressed(Key::LShift) || IsKeyPressed(Key::RShift));
    io.AddKeyEvent(ImGuiMod_Alt, IsKeyPressed(Key::LAlt) || IsKeyPressed(Key::RAlt));

    // Update buttons
    for (int32_t i = 0; i < IM_ARRAYSIZE(io.MouseDown); i++) {
        // If a mouse press event came, always pass it as "mouse held this frame", so we don't miss click-release events that are shorter than 1 frame.
        io.MouseDown[i] = m_ButtonJustPressed[i] || glfwGetMouseButton(m_Window, i) != 0;
        m_ButtonJustPressed[i] = false;
    }

    // Update mouse position
    if (glfwGetWindowAttrib(m_Window, GLFW_FOCUSED) != 0) {
        if (io.WantSetMousePos)
            glfwSetCursorPos(m_Window, (double)io.MousePos.x, (double)io.MousePos.y);
        else {
            double mouse_x, mouse_y;
            glfwGetCursorPos(m_Window, &mouse_x, &mouse_y);
            io.MousePos = ImVec2((float)mouse_x, (float)mouse_y);
        }
    }

    // Update mouse cursor
    if ((io.ConfigFlags & ImGuiConfigFlags_NoMouseCursorChange) == 0 && glfwGetInputMode(m_Window, GLFW_CURSOR) == GLFW_CURSOR_NORMAL) {
        ImGuiMouseCursor cursor = ImGui::GetMouseCursor();
        if (cursor == ImGuiMouseCursor_None || io.MouseDrawCursor) {
            // Hide OS mouse cursor if imgui is drawing it or if it wants no cursor
            CursorMode(GLFW_CURSOR_HIDDEN);
        } else {
            // Show OS mouse cursor
            glfwSetCursor(m_Window, m_MouseCursors[cursor] ? m_MouseCursors[cursor] : m_MouseCursors[ImGuiMouseCursor_Arrow]);
            CursorMode(GLFW_CURSOR_NORMAL);
        }
    }

    // Start the frame. This call will update the io.WantCaptureMouse, io.WantCaptureKeyboard flag that you can use to dispatch inputs (or not) to your application.
    ImGui::NewFrame();
}

void SampleBase::EndUI(const nri::StreamerInterface& streamerInterface, nri::Streamer& streamer) {
    if (!HasUserInterface())
        return;

    ImGui::EndFrame();
    ImGui::Render();

    const ImDrawData& drawData = *ImGui::GetDrawData();

    // Prepare
    uint32_t vertexDataSize = drawData.TotalVtxCount * sizeof(ImDrawVertOpt);
    vertexDataSize = helper::Align(vertexDataSize, 16);
    uint32_t indexDataSize = drawData.TotalIdxCount * sizeof(ImDrawIdx);
    indexDataSize = helper::Align(indexDataSize, 16);
    uint32_t totalDataSize = vertexDataSize + indexDataSize;
    if (!totalDataSize)
        return;

    if (m_UiData.size() < totalDataSize)
        m_UiData.resize(totalDataSize);

    // Repack geometry
    uint8_t* indexData = m_UiData.data();
    ImDrawVertOpt* vertexData = (ImDrawVertOpt*)(indexData + indexDataSize);

    for (int32_t n = 0; n < drawData.CmdListsCount; n++) {
        const ImDrawList& drawList = *drawData.CmdLists[n];

        for (int32_t i = 0; i < drawList.VtxBuffer.Size; i++) {
            const ImDrawVert* v = drawList.VtxBuffer.Data + i;

            ImDrawVertOpt opt;
            opt.pos[0] = v->pos.x;
            opt.pos[1] = v->pos.y;
            opt.uv = Packing::float2_to_unorm_16_16(float2(v->uv.x, v->uv.y));
            opt.col = v->col;

            memcpy(vertexData++, &opt, sizeof(opt));
        }

        size_t size = drawList.IdxBuffer.Size * sizeof(ImDrawIdx);
        memcpy(indexData, drawList.IdxBuffer.Data, size);
        indexData += size;
    }

    // Add update request
    nri::BufferUpdateRequestDesc bufferUpdateRequestDesc = {};
    bufferUpdateRequestDesc.data = m_UiData.data();
    bufferUpdateRequestDesc.dataSize = m_UiData.size();

    m_IbOffset = streamerInterface.AddStreamerBufferUpdateRequest(streamer, bufferUpdateRequestDesc);
    m_VbOffset = m_IbOffset + indexDataSize;
}

void SampleBase::RenderUI(const nri::CoreInterface& NRI, const nri::StreamerInterface& streamerInterface, nri::Streamer& streamer, nri::CommandBuffer& commandBuffer, float sdrScale, bool isSrgb) {
    if (!HasUserInterface() || m_VbOffset == m_IbOffset)
        return;

    float consts[4];
    consts[0] = 1.0f / ImGui::GetIO().DisplaySize.x;
    consts[1] = 1.0f / ImGui::GetIO().DisplaySize.y;
    consts[2] = sdrScale;
    consts[3] = isSrgb ? 1.0f : 0.0f;

    helper::Annotation annotation(NRI, commandBuffer, "UI");

    NRI.CmdSetDescriptorPool(commandBuffer, *m_DescriptorPool);
    NRI.CmdSetPipelineLayout(commandBuffer, *m_PipelineLayout);
    NRI.CmdSetPipeline(commandBuffer, *m_Pipeline);
    NRI.CmdSetRootConstants(commandBuffer, 0, consts, sizeof(consts));
    NRI.CmdSetDescriptorSet(commandBuffer, 0, *m_DescriptorSet, nullptr);

    nri::Buffer* geometryBuffer = streamerInterface.GetStreamerDynamicBuffer(streamer);
    NRI.CmdSetIndexBuffer(commandBuffer, *geometryBuffer, m_IbOffset, sizeof(ImDrawIdx) == 2 ? nri::IndexType::UINT16 : nri::IndexType::UINT32);
    NRI.CmdSetVertexBuffers(commandBuffer, 0, 1, &geometryBuffer, &m_VbOffset);

    const nri::Viewport viewport = {0.0f, 0.0f, ImGui::GetIO().DisplaySize.x, ImGui::GetIO().DisplaySize.y, 0.0f, 1.0f};
    NRI.CmdSetViewports(commandBuffer, &viewport, 1);

    const ImDrawData& drawData = *ImGui::GetDrawData();
    int32_t vertexOffset = 0;
    uint32_t indexOffset = 0;
    for (int32_t n = 0; n < drawData.CmdListsCount; n++) {
        const ImDrawList& drawList = *drawData.CmdLists[n];
        for (int32_t i = 0; i < drawList.CmdBuffer.Size; i++) {
            const ImDrawCmd& drawCmd = drawList.CmdBuffer[i];
            if (drawCmd.UserCallback)
                drawCmd.UserCallback(&drawList, &drawCmd);
            else {
                nri::Rect rect = {
                    (int16_t)drawCmd.ClipRect.x,
                    (int16_t)drawCmd.ClipRect.y,
                    (nri::Dim_t)(drawCmd.ClipRect.z - drawCmd.ClipRect.x),
                    (nri::Dim_t)(drawCmd.ClipRect.w - drawCmd.ClipRect.y)};

                if (rect.width != 0 && rect.height != 0) {
                    NRI.CmdSetScissors(commandBuffer, &rect, 1);
                    NRI.CmdDrawIndexed(commandBuffer, {drawCmd.ElemCount, 1, indexOffset, vertexOffset, 0});
                }
            }
            indexOffset += drawCmd.ElemCount;
        }
        vertexOffset += drawList.VtxBuffer.Size;
    }
}

bool SampleBase::Create(int32_t argc, char** argv, const char* windowTitle) {
    // Command line
    cmdline::parser cmdLine;

    InitCmdLineDefault(cmdLine);
    InitCmdLine(cmdLine);

    bool parseStatus = cmdLine.parse(argc, argv);

    if (cmdLine.exist("help")) {
        printf("\n%s", cmdLine.usage().c_str());
        return false;
    }

    if (!parseStatus) {
        printf("\n%s\n\n%s", cmdLine.error().c_str(), cmdLine.usage().c_str());
        return false;
    }

    ReadCmdLineDefault(cmdLine);
    ReadCmdLine(cmdLine);

    // Init GLFW
    glfwSetErrorCallback(GLFW_ErrorCallback);

    if (!glfwInit())
        return false;

    // Window size
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();

    float contentScale = 1.0f;
    if (m_DpiMode != 0) {
        float unused;
        glfwGetMonitorContentScale(monitor, &contentScale, &unused);
        printf("DPI scale %.1f%% (%s)\n", contentScale * 100.0f, m_DpiMode == 2 ? "quality" : "performance");
    }

    m_WindowResolution.x = (uint32_t)floor(m_OutputResolution.x * contentScale);
    m_WindowResolution.y = (uint32_t)floor(m_OutputResolution.y * contentScale);

    const GLFWvidmode* vidmode = glfwGetVideoMode(monitor);
    const uint32_t screenW = (uint32_t)vidmode->width;
    const uint32_t screenH = (uint32_t)vidmode->height;

    m_WindowResolution.x = min(m_WindowResolution.x, screenW);
    m_WindowResolution.y = min(m_WindowResolution.y, screenH);

    // Rendering output size
    m_OutputResolution.x = min(m_OutputResolution.x, m_WindowResolution.x);
    m_OutputResolution.y = min(m_OutputResolution.y, m_WindowResolution.y);

    if (m_DpiMode == 2)
        m_OutputResolution = m_WindowResolution;

    // Window creation
    bool decorated = m_WindowResolution.x != screenW && m_WindowResolution.y != screenH;

    printf("Creating %swindow (%u, %u)\n", decorated ? "" : "borderless ", m_WindowResolution.x, m_WindowResolution.y);

    glfwDefaultWindowHints();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_VISIBLE, 0);
    glfwWindowHint(GLFW_DECORATED, decorated ? 1 : 0);
    glfwWindowHint(GLFW_RESIZABLE, 0);

    char windowName[256];
    snprintf(windowName, sizeof(windowName), "%s [%s]", windowTitle, cmdLine.get<std::string>("api").c_str());

    m_Window = glfwCreateWindow(m_WindowResolution.x, m_WindowResolution.y, windowName, NULL, NULL);
    if (!m_Window) {
        glfwTerminate();
        return false;
    }

    int32_t x = (screenW - m_WindowResolution.x) >> 1;
    int32_t y = (screenH - m_WindowResolution.y) >> 1;
    glfwSetWindowPos(m_Window, x, y);

#if _WIN32
    m_NRIWindow.windows.hwnd = glfwGetWin32Window(m_Window);
#elif __linux__
    m_NRIWindow.x11.dpy = glfwGetX11Display();
    m_NRIWindow.x11.window = glfwGetX11Window(m_Window);
#elif __APPLE__
    m_NRIWindow.metal.caMetalLayer = GetMetalLayer(m_Window);
#endif

    // Main initialization
    printf("Loading...\n");

    nri::GraphicsAPI graphicsAPI = nri::GraphicsAPI::VK;
    if (cmdLine.get<std::string>("api") == "D3D11")
        graphicsAPI = nri::GraphicsAPI::D3D11;
    else if (cmdLine.get<std::string>("api") == "D3D12")
        graphicsAPI = nri::GraphicsAPI::D3D12;

    bool result = Initialize(graphicsAPI);

    // Set callbacks and show window
    glfwSetWindowUserPointer(m_Window, this);
    glfwSetKeyCallback(m_Window, GLFW_KeyCallback);
    glfwSetCharCallback(m_Window, GLFW_CharCallback);
    glfwSetMouseButtonCallback(m_Window, GLFW_ButtonCallback);
    glfwSetCursorPosCallback(m_Window, GLFW_CursorPosCallback);
    glfwSetScrollCallback(m_Window, GLFW_ScrollCallback);
    glfwShowWindow(m_Window);

    return result;
}

void SampleBase::RenderLoop() {
    for (uint32_t i = 0; i < m_FrameNum; i++) {
        LatencySleep(i);

        // Events
        glfwPollEvents();

        m_IsActive = glfwGetWindowAttrib(m_Window, GLFW_FOCUSED) != 0;
        if (!m_IsActive) {
            i--;
            continue;
        }

        if (glfwWindowShouldClose(m_Window) || this->AppShouldClose())
            break;

        PrepareFrame(i);
        RenderFrame(i);

        double cursorPosx, cursorPosy;
        glfwGetCursorPos(m_Window, &cursorPosx, &cursorPosy);
        m_MousePosPrev = float2(float(cursorPosx), float(cursorPosy));
        m_MouseWheel = 0.0f;
        m_MouseDelta = float2(0.0f);

        m_Timer.UpdateFrameTime();
    }

    printf(
        "FPS:\n"
        "  Last frame : %.2f fps (%.3f ms)\n"
        "  Average    : %.2f fps (%.3f ms)\n"
        "  Smoothed   : %.2f fps (%.3f ms)\n"
        "Shutting down...\n",
        1000.0f / m_Timer.GetFrameTime(), m_Timer.GetFrameTime(),
        1000.0f / m_Timer.GetSmoothedFrameTime(), m_Timer.GetSmoothedFrameTime(),
        1000.0f / m_Timer.GetVerySmoothedFrameTime(), m_Timer.GetVerySmoothedFrameTime());
}

void SampleBase::CursorMode(int32_t mode) {
    if (mode == GLFW_CURSOR_NORMAL) {
        glfwSetInputMode(m_Window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
#if defined(_WIN32)
        // GLFW works with cursor visibility incorrectly
        for (uint32_t n = 0; ::ShowCursor(1) < 0 && n < 256; n++)
            ;
#endif
    } else {
        glfwSetInputMode(m_Window, GLFW_CURSOR, mode);
#if defined(_WIN32)
        // GLFW works with cursor visibility incorrectly
        for (uint32_t n = 0; ::ShowCursor(0) >= 0 && n < 256; n++)
            ;
#endif
    }
}

void SampleBase::InitCmdLineDefault(cmdline::parser& cmdLine) {
#if _WIN32
    std::string graphicsAPI = "D3D12";
#else
    std::string graphicsAPI = "VULKAN";
#endif

    cmdLine.add("help", '?', "print this message");
    cmdLine.add<std::string>("api", 'a', "graphics API: D3D11, D3D12 or VULKAN", false, graphicsAPI, cmdline::oneof<std::string>("D3D11", "D3D12", "VULKAN"));
    cmdLine.add<std::string>("scene", 's', "scene", false, m_SceneFile);
    cmdLine.add<uint32_t>("width", 'w', "output resolution width", false, m_OutputResolution.x);
    cmdLine.add<uint32_t>("height", 'h', "output resolution height", false, m_OutputResolution.y);
    cmdLine.add<uint32_t>("frameNum", 'f', "max frames to render", false, m_FrameNum);
    cmdLine.add<uint32_t>("vsyncInterval", 'v', "vsync interval", false, m_VsyncInterval);
    cmdLine.add<uint32_t>("dpiMode", 0, "DPI mode", false, m_DpiMode);
    cmdLine.add("debugAPI", 0, "enable graphics API validation layer");
    cmdLine.add("debugNRI", 0, "enable NRI validation layer");
}

void SampleBase::ReadCmdLineDefault(cmdline::parser& cmdLine) {
    m_SceneFile = cmdLine.get<std::string>("scene");
    m_OutputResolution.x = cmdLine.get<uint32_t>("width");
    m_OutputResolution.y = cmdLine.get<uint32_t>("height");
    m_FrameNum = cmdLine.get<uint32_t>("frameNum");
    m_VsyncInterval = (uint8_t)cmdLine.get<uint32_t>("vsyncInterval");
    m_DebugAPI = cmdLine.exist("debugAPI");
    m_DebugNRI = cmdLine.exist("debugNRI");
    m_DpiMode = cmdLine.get<uint32_t>("dpiMode");
}

void SampleBase::EnableMemoryLeakDetection([[maybe_unused]] uint32_t breakOnAllocationIndex) {
#if (defined(_DEBUG) && defined(_WIN32))
    int32_t flag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
    flag |= _CRTDBG_LEAK_CHECK_DF;
    _CrtSetDbgFlag(flag);

    // https://msdn.microsoft.com/en-us/library/x98tx3cf.aspx
    if (breakOnAllocationIndex)
        _crtBreakAlloc = breakOnAllocationIndex;
#endif
}
