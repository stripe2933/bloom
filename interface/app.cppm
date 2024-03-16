export module bloom:app;

import std;
import vulkan_hpp;

export class App {
public:
    struct QueueFamilyIndices {
        std::uint32_t compute;

        explicit QueueFamilyIndices(
            vk::PhysicalDevice physicalDevice) {
            for (std::uint32_t idx = 0; vk::QueueFamilyProperties properties : physicalDevice.getQueueFamilyProperties()) {
                if (properties.queueFlags & vk::QueueFlagBits::eCompute) {
                    compute = idx;
                    return;
                }
                ++idx;
            }

            throw std::invalid_argument { "physicalDevice has no compute queue family." };
        }
    };

    struct Queues {
        vk::Queue compute;

        Queues(
            vk::Device device,
            const QueueFamilyIndices &queueFamilyIndices)
            : compute { device.getQueue(queueFamilyIndices.compute, 0) } {
        }
    };

    vk::raii::Context context;
    vk::raii::Instance instance = createInstance();
    vk::raii::PhysicalDevice physicalDevice = createPhysicalDevice();
    QueueFamilyIndices queueFamilyIndices { *physicalDevice };
    vk::raii::Device device = createDevice();
    Queues queues { *device, queueFamilyIndices };

private:
    [[nodiscard]] auto createInstance() const -> vk::raii::Instance {
        constexpr vk::ApplicationInfo appInfo {
            "Bloom", 0,
            nullptr, 0,
            vk::makeApiVersion(0, 1, 1, 0),
        };

        const std::vector<const char*> instanceLayers {
#ifndef NDEBUG
            "VK_LAYER_KHRONOS_validation"
#endif
        };
        const std::vector<const char*> instanceExtensions {
#if __APPLE__
            "VK_KHR_portability_enumeration",
            "VK_KHR_get_physical_device_properties2",
#endif
        };
        return vk::raii::Instance { context, vk::InstanceCreateInfo {
#if __APPLE__
            vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR,
#else
            {},
#endif
            &appInfo,
            instanceLayers,
            instanceExtensions,
        } };
    }

    [[nodiscard]] auto createPhysicalDevice() const -> vk::raii::PhysicalDevice {
        std::vector physicalDevices = instance.enumeratePhysicalDevices();
        auto adequatePhysicalDevices = physicalDevices | std::views::filter([](const vk::raii::PhysicalDevice &physicalDevice) {
            // Check compute queue family support.
            try {
                [[maybe_unused]] const QueueFamilyIndices queueFamilyIndices { *physicalDevice };
                return true;
            }
            catch (const std::invalid_argument&) {
                return false;
            }
        });

        // Rate physical devices based on their properties, and select the best one.
        auto it = std::ranges::max_element(adequatePhysicalDevices, {}, [](const vk::raii::PhysicalDevice &physicalDevice) {
            const vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
            return properties.limits.maxComputeWorkGroupInvocations;
        });
        if (it != std::end(adequatePhysicalDevices)) {
            return *it;
        }

        throw std::runtime_error { "No adequate physical device found." };
    }

    [[nodiscard]] auto createDevice() const -> vk::raii::Device {
        constexpr float queuePriority = 1.f;
        const vk::DeviceQueueCreateInfo queueCreateInfo {
            {},
            queueFamilyIndices.compute,
            vk::ArrayProxyNoTemporaries(queuePriority),
        };

        constexpr std::array deviceExtensions {
            "VK_KHR_portability_subset",
            "VK_EXT_shader_atomic_float",
        };
        constexpr auto physicalDeviceFeatures
            = vk::PhysicalDeviceFeatures{}
            .setShaderStorageImageWriteWithoutFormat(vk::True);
        return { physicalDevice, vk::StructureChain {
            vk::DeviceCreateInfo {
                {},
                queueCreateInfo,
                {},
                deviceExtensions,
                &physicalDeviceFeatures,
            },
            vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT{}
                .setShaderBufferFloat32AtomicAdd(vk::True),
        }.get() };
    }
};