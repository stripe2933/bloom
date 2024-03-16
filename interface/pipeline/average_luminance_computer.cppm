module;

#include <vkutil_macros.hpp>

export module bloom:pipeline.average_luminance_computer;

import std;
import vkutil;

export class AverageLuminanceComputer {
public:
    VKUTIL_NAMED_DESCRIPTOR_INFO(2, inputImageView, outputBuffer);

    DescriptorSetLayouts descriptorSetLayouts;
    vk::raii::PipelineLayout pipelineLayout;
    vk::raii::Pipeline pipeline;

    explicit AverageLuminanceComputer(
        const vk::raii::Device &device)
        : descriptorSetLayouts { std::array { createDescriptorSetLayouts(device) } },
          pipelineLayout { createPipelineLayout(device) },
          pipeline { createPipeline(device) } { }

    [[nodiscard]] static auto getWriteDescriptorSets0(
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

    [[nodiscard]] static auto getWriteDescriptorSets1(
        const DescriptorSets &descriptorSets,
        vk::DescriptorBufferInfo outputBufferInfo) noexcept
    -> vkutil::RefHolder<vk::WriteDescriptorSet, vk::DescriptorBufferInfo> {
        return {
            [&](const vk::DescriptorBufferInfo &bufferInfo) {
                return vk::WriteDescriptorSet {
                    descriptorSets.outputBuffer,
                    0,
                    0,
                    vk::DescriptorType::eStorageBuffer,
                    {},
                    bufferInfo,
                };
            },
            std::move(outputBufferInfo),
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
    [[nodiscard]] static auto createDescriptorSetLayouts(
        const vk::raii::Device &device)
    -> std::array<vk::raii::DescriptorSetLayout, 2> {
        constexpr std::tuple layouts {
            std::array {
                vk::DescriptorSetLayoutBinding {
                    0,
                    vk::DescriptorType::eStorageImage,
                    1,
                    vk::ShaderStageFlagBits::eCompute,
                },
            },
            std::array {
                vk::DescriptorSetLayoutBinding {
                    0,
                    vk::DescriptorType::eStorageBuffer,
                    1,
                    vk::ShaderStageFlagBits::eCompute,
                },
            },
        };
        return std::apply([&](const auto &...layout) {
            return std::array {
                vk::raii::DescriptorSetLayout { device, vk::DescriptorSetLayoutCreateInfo {
                    {},
                    layout,
                } }...
            };
        }, layouts);
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
        const vk::raii::ShaderModule shaderModule { device, vkutil::getShaderModuleCreateInfo("shaders/average_luminance.comp.spv") };
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
