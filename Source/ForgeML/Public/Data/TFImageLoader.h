#pragma once

#include <string>
#include <vector>

namespace cppflow
{
	class tensor;
}

namespace TF
{
	/// <summary>
	/// Enum representing the order of channels in an image tensor.
	/// </summary>
	enum class ChannelOrder
	{
		GrayScale,

		BGR,
		BGRA,
		RGB,
		RGBA
	};

	/// <summary>
	/// Enum representing the shape order of an image tensor.
	/// </summary>
	enum class ShapeOrder
	{
		WidthHeightChannels,
		HeightWidthChannels,
		ChannelsHeightWidth,
		ChannelsWidthHeight
	};

	/// <summary>
	/// Struct representing a loader for image tensors.
	/// </summary>
	struct ImageTensorLoader 
	{
	public:
		/// <summary>
		/// Constructor to initialize a ImageTensorLoader with specified parameters.
		/// </summary>
		/// <param name="width">The width of the image</param>
		/// <param name="height">The height of the image</param>
		/// <param name="channels">The number of channels of the image</param>
		/// <param name="normalize">Whether to normalize the image</param>
		/// <param name="order">The channel order of the image</param>
		/// <param name="shape">The shape order of the image</param>
		ImageTensorLoader(uint32_t width, 
						  uint32_t height, 
						  uint32_t channels,
						  bool normalize = true,
						  ChannelOrder order = ChannelOrder::RGBA,
						  ShapeOrder shape = ShapeOrder::WidthHeightChannels);

		/// <summary>
		/// Loads an image from the specified path and converts it to a tensor.
		/// </summary>
		/// <param name="image_path">The file path of the image</param>
		/// <param name="output">The output tensor</param>
		/// <returns>True whether the conversion is successful</returns>
		bool Load(const std::string& image_path, 
				  cppflow::tensor& output);
	private:
		uint32_t mWidth = 0;
		uint32_t mHeight = 0;
		uint32_t mChannels = 0;
		bool mNormalize = true;
		ChannelOrder mChannelOrder = ChannelOrder::RGBA;
		ShapeOrder mShapeOrder = ShapeOrder::WidthHeightChannels;
	};
}