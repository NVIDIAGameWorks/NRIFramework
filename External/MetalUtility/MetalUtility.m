#include "MetalUtility.h"

#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#import <QuartzCore/CAMetalLayer.h>

// Vulkan swap-chain creation requires the metal layer that was created
void* GetMetalLayer(GLFWwindow* window)
{
    if (!window)
        return NULL;

    NSWindow* nsWindow = glfwGetCocoaWindow(window);
    if (!nsWindow)
        return NULL;

    NSView* contentView = [nsWindow contentView];
    if (![contentView.layer isKindOfClass:[CAMetalLayer class]])
    {
        [contentView setLayer:[CAMetalLayer layer]];
        [contentView setWantsLayer:YES];
    }

    return (__bridge void*)[contentView layer];
}
