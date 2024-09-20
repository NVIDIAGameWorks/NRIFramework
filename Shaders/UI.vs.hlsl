// © 2021 NVIDIA Corporation

#include "NRICompatibility.hlsli"

struct PushConstants
{
    float2 gInvScreenSize;
    float gSdrScale;
    float gIsSrgb;
};

NRI_ROOT_CONSTANTS( PushConstants, g_PushConstants, 0, 0 );

struct VS_INPUT
{
    float2 pos : POSITION0;
    float2 uv : TEXCOORD0;
    float4 col : COLOR0;
};

struct PS_INPUT
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD0;
    float4 col : COLOR0;
};

PS_INPUT main( VS_INPUT input )
{
    float2 p = input.pos.xy * g_PushConstants.gInvScreenSize;
    p = p * 2.0 - 1.0;
    p.y = -p.y;

    PS_INPUT output;
    output.pos = float4( p, 0, 1 );
    output.col = input.col;
    output.uv  = input.uv;

    return output;
}