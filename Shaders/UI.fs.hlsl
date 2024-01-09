// © 2021 NVIDIA Corporation

#include "Include/BindingBridge.hlsli"
#include "STL.hlsli"

struct PushConstants
{
    float2 gInvScreenSize;
    float gSdrScale;
    float gIsSrgb;
};

NRI_PUSH_CONSTANTS( PushConstants, g_PushConstants, 0 );

struct PS_INPUT
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
    float4 col : COLOR0;
};

NRI_RESOURCE( SamplerState, sampler0, s, 0, 0 );
NRI_RESOURCE( Texture2D<float>, texture0, t, 0, 0 );

float4 main( PS_INPUT input ) : SV_Target
{
    float4 color = input.col;
    color.w *= texture0.Sample( sampler0, input.uv );

    if( g_PushConstants.gIsSrgb == 0.0 )
        color.xyz = STL::Color::FromSrgb( color.xyz );

    color.xyz *= g_PushConstants.gSdrScale;

    return color;
}