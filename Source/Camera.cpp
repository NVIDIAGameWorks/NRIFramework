// © 2021 NVIDIA Corporation

#include "NRIFramework.h"

void Camera::Initialize(const float3& position, const float3& lookAt, bool isRelative)
{
    float3 dir = normalize(lookAt - position);

    float3 rot;
    rot.x = atan2(dir.y, dir.x);
    rot.y = asin(dir.z);
    rot.z = 0.0f;

    state.globalPosition = double3(position);
    state.rotation = degrees(rot);
    m_IsRelative = isRelative;
}

void Camera::InitializeWithRotation(const float3& position, const float3& rotation, bool isRelative)
{
    state.globalPosition = double3(position);
    state.rotation = rotation;
    m_IsRelative = isRelative;
}

void Camera::Update(const CameraDesc& desc, uint32_t frameIndex)
{
    uint32_t projFlags = desc.isReversedZ ? PROJ_REVERSED_Z : 0;
    projFlags |= desc.isPositiveZ ? PROJ_LEFT_HANDED : 0;

    // Position
    const float3 vRight = state.mWorldToView.GetRow0().xyz;
    const float3 vUp = state.mWorldToView.GetRow1().xyz;
    const float3 vForward = state.mWorldToView.GetRow2().xyz;

    float3 delta = desc.dLocal * desc.timeScale;
    delta.z *= desc.isPositiveZ ? 1.0f : -1.0f;

    state.globalPosition += double3(vRight * delta.x);
    state.globalPosition += double3(vUp * delta.y);
    state.globalPosition += double3(vForward * delta.z);
    state.globalPosition += double3(desc.dUser);

    if (desc.limits.IsValid())
        state.globalPosition = clamp(state.globalPosition, double3(desc.limits.vMin), double3(desc.limits.vMax));

    if (desc.isCustomMatrixSet)
    {
        const float3 vCustomRight = desc.customMatrix.GetRow3().xyz;
        state.globalPosition = double3(vCustomRight);
    }

    if (m_IsRelative)
    {
        state.position = float3::Zero();
        statePrev.position = float3(statePrev.globalPosition - state.globalPosition);
        statePrev.mWorldToView.PreTranslation(-statePrev.position);
    }
    else
    {
        state.position = float3(state.globalPosition);
        statePrev.position = float3(statePrev.globalPosition);
    }

    // Rotation
    float angularSpeed = 0.03f * saturate( desc.horizontalFov * 0.5f / 90.0f );

    state.rotation.x += desc.dYaw * angularSpeed;
    state.rotation.y += desc.dPitch * angularSpeed;

    state.rotation.x = fmodf(state.rotation.x, 360.0f);
    state.rotation.y = clamp(state.rotation.y, -90.0f, 90.0f);

    if (desc.isCustomMatrixSet)
    {
        state.mViewToWorld = desc.customMatrix;

        state.rotation = degrees( state.mViewToWorld.GetRotationYPR() );
        state.rotation.z = 0.0f;
    }
    else
        state.mViewToWorld.SetupByRotationYPR( radians(state.rotation.x), radians(state.rotation.y), radians(state.rotation.z) );

    state.mWorldToView = state.mViewToWorld;
    state.mWorldToView.PreTranslation( float3(state.mWorldToView.GetRow2().xyz) * desc.backwardOffset );
    state.mWorldToView.WorldToView(projFlags);
    state.mWorldToView.PreTranslation( -state.position );

    // Projection
    if(desc.orthoRange > 0.0f)
    {
        float x = desc.orthoRange;
        float y = desc.orthoRange / desc.aspectRatio;
        state.mViewToClip.SetupByOrthoProjection(-x, x, -y, y, desc.nearZ, desc.farZ, projFlags);
    }
    else
    {
        if (desc.farZ == 0.0f)
            state.mViewToClip.SetupByHalfFovxInf(0.5f * radians(desc.horizontalFov), desc.aspectRatio, desc.nearZ, projFlags);
        else
            state.mViewToClip.SetupByHalfFovx(0.5f * radians(desc.horizontalFov), desc.aspectRatio, desc.nearZ, desc.farZ, projFlags);
    }

    // Other
    state.mWorldToClip = state.mViewToClip * state.mWorldToView;

    state.mViewToWorld = state.mWorldToView;
    state.mViewToWorld.InvertOrtho();

    state.mClipToView = state.mViewToClip;
    state.mClipToView.Invert();

    state.mClipToWorld = state.mWorldToClip;
    state.mClipToWorld.Invert();

    state.viewportJitter = Sequence::Halton2D( frameIndex ) - 0.5f;

    // Previous other
    statePrev.mWorldToClip = statePrev.mViewToClip * statePrev.mWorldToView;

    statePrev.mViewToWorld = statePrev.mWorldToView;
    statePrev.mViewToWorld.InvertOrtho();

    statePrev.mClipToWorld = statePrev.mWorldToClip;
    statePrev.mClipToWorld.Invert();
}
