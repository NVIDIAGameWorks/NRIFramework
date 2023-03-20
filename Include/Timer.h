#pragma once

#include <cstdint>

class Timer
{
public:
    Timer();

    void UpdateFrameTime();

    // In milliseconds
    double GetTimeStamp() const;
    
    inline double GetLastFrameTimeStamp() const
    { return m_Time * m_InvTicksPerMs; }

    inline float GetFrameTime() const
    { return m_Delta; }

    inline float GetSmoothedFrameTime() const
    { return m_SmoothedDelta; }

    inline float GetVerySmoothedFrameTime() const
    { return m_VerySmoothedDelta; }

private:

private:
    uint64_t m_Time = 0;
    double m_InvTicksPerMs = 0.0;
    float m_Delta = 1.0f;
    float m_SmoothedDelta = 1.0f;
    float m_VerySmoothedDelta = 1.0f;
};