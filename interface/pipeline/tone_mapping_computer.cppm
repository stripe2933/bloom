module;

#include <vkutil_macros.hpp>

export module bloom:pipeline.tone_mapping_computer;

import std;
import vkutil;

export class ToneMappingComputer {
public:
    VKUTIL_NAMED_DESCRIPTOR_INFO(1, inputImageView);

    struct PushConstantData{
        float averageLuminance;
    };

    DescriptorSetLayouts descriptorSetLayouts;
    vk::raii::PipelineLayout pipelineLayout;
    vk::raii::Pipeline pipeline;

    explicit ToneMappingComputer(
        const vk::raii::Device &device)
        : descriptorSetLayouts { std::array { createDescriptorSetLayout(device) } },
          pipelineLayout { createPipelineLayout(device) },
          pipeline { createPipeline(device) } { }

    [[nodiscard]] static auto getWriteDescriptorSets(
        const DescriptorSets &descriptorSets,
        vk::ImageView inputImageView) noexcept
    -> vkutil::RefHolder<vk::WriteDescriptorSet, vk::DescriptorImageInfo> {
        return {
            [&](const vk::DescriptorImageInfo &inputImageInfo) {
                return vk::WriteDescriptorSet {
                    descriptorSets.inputImageView,
                    0,
                    0,
                    vk::DescriptorType::eStorageImage,
                    inputImageInfo,
                };
            },
            vk::DescriptorImageInfo {
                {},
                inputImageView,
                vk::ImageLayout::eGeneral,
            },
        };
    }

    [[nodiscard]] static auto getWorkgroupCount(
        vk::Extent2D inputImageExtent) noexcept
    -> std::array<std::uint32_t, 3> {
        constexpr vk::Extent3D LOCAL_WORKGROUP_SIZE { 16, 16, 1 };
        return {
            (inputImageExtent.width + LOCAL_WORKGROUP_SIZE.width - 1U) / LOCAL_WORKGROUP_SIZE.width,
            (inputImageExtent.height + LOCAL_WORKGROUP_SIZE.height - 1U) / LOCAL_WORKGROUP_SIZE.height,
            1,
        };
    }

private:
    [[nodiscard]] static auto createDescriptorSetLayout(
        const vk::raii::Device &device)
    -> vk::raii::DescriptorSetLayout {
        constexpr std::array layoutBindings {
            vk::DescriptorSetLayoutBinding {
                0,
                vk::DescriptorType::eStorageImage,
                1,
                vk::ShaderStageFlagBits::eCompute,
            },
        };

        return vk::raii::DescriptorSetLayout { device, vk::DescriptorSetLayoutCreateInfo {
            {},
            layoutBindings,
        } };
    }

    [[nodiscard]] auto createPipelineLayout(
        const vk::raii::Device &device) const
    -> vk::raii::PipelineLayout {
        constexpr vk::PushConstantRange pushConstantRange {
            vk::ShaderStageFlagBits::eCompute,
            0,
            sizeof(PushConstantData),
        };
        return { device, vk::PipelineLayoutCreateInfo {
            {},
            descriptorSetLayouts,
            pushConstantRange,
        } };
    }

    [[nodiscard]] auto createPipeline(
        const vk::raii::Device &device) const
    -> vk::raii::Pipeline {
        const vk::raii::ShaderModule shaderModule { device, vkutil::getShaderModuleCreateInfo("shaders/tone_mapping.comp.spv") };
        return { device, nullptr, vk::ComputePipelineCreateInfo {
            {},
            vk::PipelineShaderStageCreateInfo {
                {},
                vk::ShaderStageFlagBits::eCompute,
                *shaderModule,
                "main",
            },
            *pipelineLayout,
        } };
    }
};
