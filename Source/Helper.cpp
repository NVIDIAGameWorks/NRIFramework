#include "NRI.h"
#include "Extensions/NRIDeviceCreation.h"
#include "Helper.h"

bool helper::FindPhysicalDeviceGroup(nri::PhysicalDeviceGroup& physicalDeviceGroup)
{
    uint32_t deviceGroupNum = 0;
    nri::Result result = nri::GetPhysicalDevices(nullptr, deviceGroupNum);

    if (deviceGroupNum == 0 || result != nri::Result::SUCCESS)
        return false;

    std::vector<nri::PhysicalDeviceGroup> groups(deviceGroupNum);
    result = nri::GetPhysicalDevices(groups.data(), deviceGroupNum);

    if (result != nri::Result::SUCCESS)
        return false;

    size_t groupIndex = 0;
    for (; groupIndex < groups.size(); groupIndex++)
    {
        if (groups[groupIndex].type != nri::PhysicalDeviceType::INTEGRATED)
            break;
    }

    if (groupIndex == groups.size())
        groupIndex = 0;

    physicalDeviceGroup = groups[groupIndex];

    return true;
}