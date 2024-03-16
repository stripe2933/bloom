#include <stb_image_write.h>

import std;
import bloom;
import glm;
import vkutil;

class BloomApp : App {
public:
    vkutil::DeviceMemoryAllocator allocator { *physicalDevice };
    vk::raii::DescriptorPool descriptorPool = createDescriptorPool();
    vk::raii::CommandPool computeCommandPool = createCommandPool(queueFamilyIndices.compute);

    void run() const {
        const vkutil::AllocatedImage diamondImage = [&] {
            vkutil::AllocatedImage result { device, allocator, vk::ImageCreateInfo {
                {},
                vk::ImageType::e2D,
                vk::Format::eR16G16B16A16Sfloat,
                vk::Extent3D { 1920, 1080, 1 },
                6, 1,
                vk::SampleCountFlagBits::e1,
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc,
            }, vk::MemoryPropertyFlagBits::eDeviceLocal };

            const DiamondImageComputer diamondImageComputer { device };
            const DiamondImageComputer::DescriptorSets descriptorSets { *device, *descriptorPool, diamondImageComputer.descriptorSetLayouts };
            const vk::raii::ImageView imageView { device, vk::ImageViewCreateInfo {
                {},
                result,
                vk::ImageViewType::e2D,
                result.format,
                {},
                { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 },
            } };
            (*device).updateDescriptorSets(diamondImageComputer.getWriteDescriptorSets(descriptorSets, *imageView).get(), {});

            vkutil::executeSingleCommand(*device, *computeCommandPool, queues.compute, [&](vk::CommandBuffer commandBuffer) {
                // Transition image layout to eGeneral.
                const vk::ImageMemoryBarrier imageMemoryBarrier {
                    {}, vk::AccessFlagBits::eShaderWrite,
                    {}, vk::ImageLayout::eGeneral,
                    {}, {},
                    result,
                    { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 },
                };
                commandBuffer.pipelineBarrier(
                    vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader,
                    {}, {}, {}, imageMemoryBarrier);

                // Dispatch compute shader.
                commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *diamondImageComputer.pipeline);
                commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *diamondImageComputer.pipelineLayout, 0, descriptorSets, {});
                const std::array workgroupCount = DiamondImageComputer::getWorkgroupCount(vk::Extent2D { diamondImage.extent.width, diamondImage.extent.height });
                commandBuffer.dispatch(get<0>(workgroupCount), get<1>(workgroupCount), get<2>(workgroupCount));
            });

            return result;
        }();

        {
            const BloomDownsampleComputer bloomDownsampleComputer { device };
            const BloomUpsampleComputer bloomUpsampleComputer { device };
            const std::vector<vk::raii::ImageView> mipViews { std::from_range, std::views::iota(0U, diamondImage.mipLevels) | std::views::transform([&](std::uint32_t mipLevel) {
                return vk::raii::ImageView { device, vk::ImageViewCreateInfo {
                    {},
                    diamondImage,
                    vk::ImageViewType::e2D,
                    diamondImage.format,
                    {},
                    { vk::ImageAspectFlagBits::eColor, mipLevel, 1, 0, 1 },
                } };
            }) };

            vkutil::executeSingleCommand(*device, *computeCommandPool, queues.compute, [&](vk::CommandBuffer commandBuffer) {
                // Downsample phase.
                commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *bloomDownsampleComputer.pipeline);
                for (auto [srcLevel, dstLevel] : std::views::iota(0U, diamondImage.mipLevels) | utils::views::pairwise) {
                    // Transition diamondImage[mipLevel=srcLevel] layout to eShaderReadOnlyOptimal, and diamondImage[mipLevel=outputLevel] layout to eGeneral.
                    const std::array barriers {
                        vk::ImageMemoryBarrier {
                            srcLevel == 0 ? vk::AccessFlagBits::eNone : vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
                            vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal,
                            {}, {},
                            diamondImage,
                            { vk::ImageAspectFlagBits::eColor, srcLevel, 1, 0, 1 },
                        },
                        vk::ImageMemoryBarrier {
                            {}, vk::AccessFlagBits::eShaderWrite,
                            {}, vk::ImageLayout::eGeneral,
                            {}, {},
                            diamondImage,
                            { vk::ImageAspectFlagBits::eColor, dstLevel, 1, 0, 1 },
                        },
                    };
                    commandBuffer.pipelineBarrier(
                        srcLevel == 0 ? vk::PipelineStageFlagBits::eTopOfPipe : vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
                        {}, {}, {}, barriers);

                    const BloomDownsampleComputer::DescriptorSets descriptorSets { *device, *descriptorPool, bloomDownsampleComputer.descriptorSetLayouts };
                    device.updateDescriptorSets(bloomDownsampleComputer.getWriteDescriptorSets(descriptorSets, *mipViews[srcLevel], *mipViews[dstLevel]).get(), {});

                    // Dispatch compute shader.
                    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *bloomDownsampleComputer.pipelineLayout, 0, descriptorSets, {});
                    commandBuffer.pushConstants<BloomDownsampleComputer::PushConstantData>(*bloomDownsampleComputer.pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, BloomDownsampleComputer::PushConstantData {
                        glm::u32vec2 { diamondImage.extent.width >> srcLevel, diamondImage.extent.height >> srcLevel },
                    });
                    const std::array workgroupCount = BloomDownsampleComputer::getWorkgroupCount(vk::Extent2D { diamondImage.extent.width >> dstLevel, diamondImage.extent.height >> dstLevel });
                    commandBuffer.dispatch(get<0>(workgroupCount), get<1>(workgroupCount), get<2>(workgroupCount));
                }

                // Upsample phase.
                commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *bloomUpsampleComputer.pipeline);
                for (auto [srcLevel, dstLevel] : std::views::iota(0U, diamondImage.mipLevels) | std::views::reverse | utils::views::pairwise) {
                    // Transition diamondImage[mipLevel=srcLevel] layout to eShaderReadOnlyOptimal, and diamondImage[mipLevel=outputLevel] layout to eGeneral.
                    const std::array barriers {
                        vk::ImageMemoryBarrier {
                            vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
                            vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal,
                            {}, {},
                            diamondImage,
                            { vk::ImageAspectFlagBits::eColor, srcLevel, 1, 0, 1 },
                        },
                        vk::ImageMemoryBarrier {
                            {}, vk::AccessFlagBits::eShaderWrite,
                            vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eGeneral,
                            {}, {},
                            diamondImage,
                            { vk::ImageAspectFlagBits::eColor, dstLevel, 1, 0, 1 },
                        },
                    };
                    commandBuffer.pipelineBarrier(
                        vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader,
                        {}, {}, {}, barriers);

                    const BloomUpsampleComputer::DescriptorSets descriptorSets { *device, *descriptorPool, bloomUpsampleComputer.descriptorSetLayouts };
                    device.updateDescriptorSets(bloomUpsampleComputer.getWriteDescriptorSets(descriptorSets, *mipViews[srcLevel], *mipViews[dstLevel]).get(), {});

                    // Dispatch compute shader.
                    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *bloomUpsampleComputer.pipelineLayout, 0, descriptorSets, {});
                    commandBuffer.pushConstants<BloomUpsampleComputer::PushConstantData>(*bloomUpsampleComputer.pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, BloomUpsampleComputer::PushConstantData {
                        glm::u32vec2 { diamondImage.extent.width >> srcLevel, diamondImage.extent.height >> srcLevel },
                    });
                    const std::array workgroupCount = BloomUpsampleComputer::getWorkgroupCount(vk::Extent2D { diamondImage.extent.width >> dstLevel, diamondImage.extent.height >> dstLevel });
                    commandBuffer.dispatch(get<0>(workgroupCount), get<1>(workgroupCount), get<2>(workgroupCount));
                }
            });
        }

        const float averageLuminance = [&]{
            const AverageLuminanceComputer averageLuminanceComputer { device };
            const AverageLuminanceComputer::DescriptorSets descriptorSets { *device, *descriptorPool, averageLuminanceComputer.descriptorSetLayouts };
            const vk::raii::ImageView diamondImageView { device, vk::ImageViewCreateInfo {
                {},
                diamondImage,
                vk::ImageViewType::e2D,
                diamondImage.format,
                {},
                { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 },
            } };
            const std::array workgroupCount = AverageLuminanceComputer::getWorkgroupCount(vk::Extent2D { diamondImage.extent.width, diamondImage.extent.height });
            const vkutil::PersistentMappedBuffer outputBuffer { device, allocator, vk::BufferCreateInfo {
                {},
                sizeof(float) * get<0>(workgroupCount) * get<1>(workgroupCount),
                vk::BufferUsageFlagBits::eStorageBuffer,
            }, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eDeviceLocal };
            (*device).updateDescriptorSets(std::array {
                averageLuminanceComputer.getWriteDescriptorSets0(descriptorSets, *diamondImageView).get(),
                averageLuminanceComputer.getWriteDescriptorSets1(descriptorSets, { outputBuffer, 0, vk::WholeSize }).get()
            }, {});

            vkutil::executeSingleCommand(*device, *computeCommandPool, queues.compute, [&](vk::CommandBuffer commandBuffer) {
                commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *averageLuminanceComputer.pipeline);
                commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *averageLuminanceComputer.pipelineLayout, 0, descriptorSets, {});
                commandBuffer.dispatch(get<0>(workgroupCount), get<1>(workgroupCount), get<2>(workgroupCount));
            });

            const std::span workgroupLuminanceSum { reinterpret_cast<const float*>(outputBuffer.data), get<0>(workgroupCount) * get<1>(workgroupCount) };
            return std::reduce(workgroupLuminanceSum.begin(), workgroupLuminanceSum.end(), 0.0) / (diamondImage.extent.width * diamondImage.extent.height);
        }();

        // Tone mapping using average luminance.
        {
            const ToneMappingComputer toneMappingComputer { device };
            const ToneMappingComputer::DescriptorSets descriptorSets { *device, *descriptorPool, toneMappingComputer.descriptorSetLayouts };
            const vk::raii::ImageView diamondImageView { device, vk::ImageViewCreateInfo {
                {},
                diamondImage,
                vk::ImageViewType::e2D,
                diamondImage.format,
                {},
                { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 },
            } };
            (*device).updateDescriptorSets(toneMappingComputer.getWriteDescriptorSets(descriptorSets, *diamondImageView).get(), {});

            vkutil::executeSingleCommand(*device, *computeCommandPool, queues.compute, [&](vk::CommandBuffer commandBuffer) {
                commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *toneMappingComputer.pipeline);
                commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *toneMappingComputer.pipelineLayout, 0, descriptorSets, {});
                commandBuffer.pushConstants<ToneMappingComputer::PushConstantData>(*toneMappingComputer.pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, ToneMappingComputer::PushConstantData {
                    averageLuminance,
                });
                const std::array workgroupCount = ToneMappingComputer::getWorkgroupCount(vk::Extent2D { diamondImage.extent.width, diamondImage.extent.height });
                commandBuffer.dispatch(get<0>(workgroupCount), get<1>(workgroupCount), get<2>(workgroupCount));
            });
        }

        // Destage image to host-visible memory.
        const vkutil::PersistentMappedBuffer destagedImageBuffer = [&] {
            vkutil::AllocatedImage _8bitImage { device, allocator, vk::ImageCreateInfo {
                {},
                vk::ImageType::e2D,
                vk::Format::eR8G8B8A8Unorm,
                vk::Extent3D { diamondImage.extent.width, diamondImage.extent.height, 1 },
                1, 1,
                vk::SampleCountFlagBits::e1,
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
            }, vk::MemoryPropertyFlagBits::eDeviceLocal };
            vkutil::PersistentMappedBuffer result { device, allocator, vk::BufferCreateInfo {
                {},
                blockSize(_8bitImage.format) * _8bitImage.extent.width * _8bitImage.extent.height,
                vk::BufferUsageFlagBits::eTransferDst,
            }, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent };

            vkutil::executeSingleCommand(*device, *computeCommandPool, queues.compute, [&](vk::CommandBuffer commandBuffer) {
                const std::array barriers {
                    vk::ImageMemoryBarrier {
                        {}, vk::AccessFlagBits::eTransferRead,
                        vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal,
                        {}, {},
                        diamondImage,
                        { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 },
                    },
                    vk::ImageMemoryBarrier {
                        {}, vk::AccessFlagBits::eTransferRead,
                        {}, vk::ImageLayout::eTransferDstOptimal,
                        {}, {},
                        _8bitImage,
                        { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 },
                    },
                };
                commandBuffer.pipelineBarrier(
                    vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
                    {}, {}, {}, barriers);

                commandBuffer.blitImage(
                    diamondImage, vk::ImageLayout::eTransferSrcOptimal,
                    _8bitImage, vk::ImageLayout::eTransferDstOptimal,
                    vk::ImageBlit {
                        { vk::ImageAspectFlagBits::eColor, 0, 0, 1 },
                        std::array { vk::Offset3D { 0, 0, 0 }, vk::Offset3D { static_cast<int>(diamondImage.extent.width), static_cast<int>(diamondImage.extent.height), 1 } },
                        { vk::ImageAspectFlagBits::eColor, 0, 0, 1 },
                        std::array { vk::Offset3D { 0, 0, 0 }, vk::Offset3D { static_cast<int>(_8bitImage.extent.width), static_cast<int>(_8bitImage.extent.height), 1 } },
                    }, vk::Filter::eLinear);

                const vk::ImageMemoryBarrier barrier {
                    vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eTransferRead,
                    vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferSrcOptimal,
                    {}, {},
                    _8bitImage,
                    { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 },
                };
                commandBuffer.pipelineBarrier(
                    vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
                    {}, {}, {}, barrier);

                commandBuffer.copyImageToBuffer(
                    _8bitImage, vk::ImageLayout::eTransferSrcOptimal,
                    result, vk::BufferImageCopy {
                        0, 0, 0,
                        { vk::ImageAspectFlagBits::eColor, 0, 0, 1 },
                        { 0, 0, 0 },
                        diamondImage.extent,
                    });
            });

            return result;
        }();

        stbi_write_png("result.png", diamondImage.extent.width, diamondImage.extent.height, 4, destagedImageBuffer.data, sizeof(glm::u8vec4) * diamondImage.extent.width);
    }

private:
    [[nodiscard]] auto createDescriptorPool() const -> vk::raii::DescriptorPool {
        constexpr std::array poolSizes {
            vk::DescriptorPoolSize {
                vk::DescriptorType::eCombinedImageSampler,
                64,
            },
            vk::DescriptorPoolSize {
                vk::DescriptorType::eStorageImage,
                64,
            },
            vk::DescriptorPoolSize {
                vk::DescriptorType::eStorageBuffer,
                64,
            },
        };
        return { device, vk::DescriptorPoolCreateInfo {
            {},
            64,
            poolSizes,
        } };
    }

    [[nodiscard]] auto createCommandPool(
        std::uint32_t queueFamilyIndex) const
    -> vk::raii::CommandPool {
        return { device, vk::CommandPoolCreateInfo {
            {},
            queueFamilyIndex,
        } };
    }
};

int main() {
    BloomApp{}.run();
}
