module;

#include <vkutil_macros.hpp>

export module bloom:pipeline.bloom_downsample_computer;

import std;
import glm;
import vkutil;

export class BloomDownsampleComputer {
public:
    VKUTIL_NAMED_DESCRIPTOR_INFO(1, inoutMipViews);

    struct PushConstantData {
        glm::u32vec2 inputImageSize;
    };

    DescriptorSetLayouts descriptorSetLayouts;
    vk::raii::PipelineLayout pipelineLayout;
    vk::raii::Pipeline pipeline;

    explicit BloomDownsampleComputer(
        const vk::raii::Device &device)
        : descriptorSetLayouts { std::array { createDescriptorSetLayout(device) } },
          pipelineLayout { createPipelineLayout(device) },
          pipeline { createPipeline(device) },
          inputImageSampler { createSampler(device) } { }

    [[nodiscard]] auto getWriteDescriptorSets(
        const DescriptorSets &descriptorSets,
        vk::ImageView inputMipView,
        vk::ImageView outputMipView) const noexcept
    -> vkutil::RefHolder<std::array<vk::WriteDescriptorSet, 2>, vk::DescriptorImageInfo, vk::DescriptorImageInfo> {
        return {
            [&](const vk::DescriptorImageInfo &inputImageInfo, const vk::DescriptorImageInfo &outputImageInfo) {
                return std::array {
                    vk::WriteDescriptorSet {
                        descriptorSets.inoutMipViews,
                        0,
                        0,
                        vk::DescriptorType::eCombinedImageSampler,
                        inputImageInfo,
                    },
                    vk::WriteDescriptorSet {
                        descriptorSets.inoutMipViews,
                        1,
                        0,
                        vk::DescriptorType::eStorageImage,
                        outputImageInfo,
                    },
                };
            },
            vk::DescriptorImageInfo {
                *inputImageSampler,
                inputMipView,
                vk::ImageLayout::eShaderReadOnlyOptimal,
            },
            vk::DescriptorImageInfo {
                {},
                outputMipView,
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
    vk::raii::Sampler inputImageSampler;

    [[nodiscard]] static auto createDescriptorSetLayout(
        const vk::raii::Device &device)
    -> vk::raii::DescriptorSetLayout {
        constexpr std::array layoutBindings {
            vk::DescriptorSetLayoutBinding {
                0,
                vk::DescriptorType::eCombinedImageSampler,
                1,
                vk::ShaderStageFlagBits::eCompute,
            },
            vk::DescriptorSetLayoutBinding {
                1,
                vk::DescriptorType::eStorageImage,
                1,
                vk::ShaderStageFlagBits::eCompute,
            },
        };
        return { device, vk::DescriptorSetLayoutCreateInfo {
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
        const vk::raii::ShaderModule shaderModule { device, vkutil::getShaderModuleCreateInfo("shaders/bloom_downsample.comp.spv") };
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

    [[nodiscard]] static auto createSampler(
        const vk::raii::Device &device)
    -> vk::raii::Sampler {
        return { device, vk::SamplerCreateInfo {
            {},
            vk::Filter::eLinear, vk::Filter::eLinear,
            {},
            vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge, {},
        } };
    }
};
