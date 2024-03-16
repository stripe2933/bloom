module;

#include <vkutil_macros.hpp>

export module bloom:pipeline.diamond_image_computer;

import std;
import vkutil;

export class DiamondImageComputer {
public:
    VKUTIL_NAMED_DESCRIPTOR_INFO(1, outputImage);

    DescriptorSetLayouts descriptorSetLayouts;
    vk::raii::PipelineLayout pipelineLayout;
    vk::raii::Pipeline pipeline;

    explicit DiamondImageComputer(
        const vk::raii::Device &device)
        : descriptorSetLayouts { std::array { createDescriptorSetLayout(device) } },
          pipelineLayout { createPipelineLayout(device) },
          pipeline { createPipeline(device) } { }

    [[nodiscard]] static auto getWriteDescriptorSets(
        const DescriptorSets &descriptorSets,
        vk::ImageView outputImageView) noexcept
    -> vkutil::RefHolder<vk::WriteDescriptorSet, vk::DescriptorImageInfo> {
        return {
            [&](const vk::DescriptorImageInfo &outputImageInfo) {
                return vk::WriteDescriptorSet {
                    descriptorSets.outputImage,
                    0,
                    0,
                    vk::DescriptorType::eStorageImage,
                    outputImageInfo,
                };
            },
            vk::DescriptorImageInfo {
                {},
                outputImageView,
                vk::ImageLayout::eGeneral,
            },
        };
    }

    [[nodiscard]] static auto getWorkgroupCount(
        vk::Extent2D imageExtent) noexcept
    -> std::array<std::uint32_t, 3> {
        constexpr vk::Extent3D LOCAL_WORKGROUP_SIZE { 16, 16, 1 };
        return {
            (imageExtent.width + LOCAL_WORKGROUP_SIZE.width - 1U) / LOCAL_WORKGROUP_SIZE.width,
            (imageExtent.height + LOCAL_WORKGROUP_SIZE.height - 1U) / LOCAL_WORKGROUP_SIZE.height,
            1,
        };
    }

private:
    [[nodiscard]] static auto createDescriptorSetLayout(
        const vk::raii::Device &device)
    -> vk::raii::DescriptorSetLayout {
        constexpr vk::DescriptorSetLayoutBinding layoutBinding {
            0,
            vk::DescriptorType::eStorageImage,
            1,
            vk::ShaderStageFlagBits::eCompute,
        };
        return { device, vk::DescriptorSetLayoutCreateInfo {
            {},
            layoutBinding,
        } };
    }

    [[nodiscard]] auto createPipelineLayout(
        const vk::raii::Device &device) const
    -> vk::raii::PipelineLayout {
        return { device, vk::PipelineLayoutCreateInfo {
            {},
            descriptorSetLayouts,
        } };
    }

    [[nodiscard]] auto createPipeline(
        const vk::raii::Device &device) const
    -> vk::raii::Pipeline {
        const vk::raii::ShaderModule shaderModule { device, vkutil::getShaderModuleCreateInfo("shaders/diamond_image.comp.spv") };
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
