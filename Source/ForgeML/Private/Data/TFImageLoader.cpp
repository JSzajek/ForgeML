#include "Data/TFImageLoader.h"

#include "CppFlowLib.h"
#include "OpenCVLib.h"

namespace TF
{
	bool ConvertToChannels(cv::Mat& img, 
						   uint32_t desired_channels, 
						   bool to_rgb = true) 
	{
		int current_channels = img.channels();

		// If already matches, no conversion needed
		if (current_channels == desired_channels) 
			return true;

		int code = -1;

		// Define dynamic conversion logic
		switch (current_channels) 
		{
			case 1:
			{
				if (desired_channels == 3)
					code = to_rgb ? cv::COLOR_GRAY2RGB : cv::COLOR_GRAY2BGR;
				else if (desired_channels == 4)
					code = to_rgb ? cv::COLOR_GRAY2RGBA : cv::COLOR_GRAY2BGRA;
				break;
			}
			case 3:
			{
				if (desired_channels == 1)
					code = to_rgb ? cv::COLOR_RGB2GRAY : cv::COLOR_BGR2GRAY;
				else if (desired_channels == 4)
					code = to_rgb ? cv::COLOR_RGB2RGBA : cv::COLOR_BGR2BGRA;
				else if (desired_channels == 3 && !to_rgb)
					code = cv::COLOR_RGB2BGR;  // fallback
				break;
			}
			case 4:
			{
				if (desired_channels == 1)
					code = to_rgb ? cv::COLOR_RGBA2GRAY : cv::COLOR_BGRA2GRAY;
				else if (desired_channels == 3)
					code = to_rgb ? cv::COLOR_RGBA2RGB : cv::COLOR_BGRA2BGR;
				break;
			}
		}

		if (code == -1)
		{
			std::cerr << "Unsupported channel conversion: from "  
					  << std::to_string(current_channels) << " to "
					  << std::to_string(desired_channels);
			return false;
		}

		cv::cvtColor(img, img, code);
		return true;
	}

	ImageTensorLoader::ImageTensorLoader(uint32_t width, 
										 uint32_t height, 
										 uint32_t channels, 
										 bool normalize, 
										 ChannelOrder order, 
										 ShapeOrder shape)
		: mWidth(width),
		mHeight(height),
		mChannels(channels),
		mNormalize(normalize),
		mChannelOrder(order),
		mShapeOrder(shape)
	{
		if (mChannels < 1 || mChannels > 4)
		{
			throw std::invalid_argument("Invalid number of channels. Must be between 1 and 4.");
		}

		if (mWidth == 0 || mHeight == 0)
		{
			throw std::invalid_argument("Width and Height must be greater than zero.");
		}

		if (mChannelOrder != ChannelOrder::GrayScale && 
			mChannelOrder != ChannelOrder::BGR && mChannelOrder != ChannelOrder::BGRA &&
			mChannelOrder != ChannelOrder::RGB && mChannelOrder != ChannelOrder::RGBA)
		{
			throw std::invalid_argument("Invalid channel order specified.");
	}
	}

	bool ImageTensorLoader::Load(const std::string& image_path, cppflow::tensor& output)
	{
		cv::Mat image = cv::imread(image_path, cv::IMREAD_UNCHANGED);
		if (image.empty())
		{
			std::cerr << "Failed to load image: " << image_path << std::endl;
			return false;
		}

		bool isGrayScale = mChannelOrder == ChannelOrder::GrayScale;

		ConvertToChannels(image, mChannels, !isGrayScale);


		cv::resize(image, image, cv::Size(mWidth, mHeight));
		if (mNormalize)
		{
			image.convertTo(image, CV_32FC(mChannels), 1.0 / 255.0);
		}

		std::vector<float> input_data;
		const auto InputPixel = [&](uint32_t x, uint32_t y, uint32_t c) -> bool
		{
			switch (mChannelOrder)
			{
				case ChannelOrder::GrayScale:
					input_data.push_back(image.at<float>(y, x));
					return true;
				case ChannelOrder::BGR:
				case ChannelOrder::RGB:
					input_data.push_back(image.at<cv::Vec3f>(y, x)[c]);
					return true;
				case ChannelOrder::BGRA:
				case ChannelOrder::RGBA:
					input_data.push_back(image.at<cv::Vec4f>(y, x)[c]);
					return true;
				default:
					std::cerr << "Unsupported Channel Order." << std::endl;
					return false;
			}
		};

		switch (mShapeOrder)
		{
			case ShapeOrder::HeightWidthChannels:
			{
				for (uint32_t y = 0; y < mHeight; ++y)
					for (uint32_t x = 0; x < mWidth; ++x)
						for (uint32_t c = 0; c < mChannels; ++c)
							if (!InputPixel(x, y, c))
								continue;
				break;
			}
			case ShapeOrder::WidthHeightChannels:
			{
				for (uint32_t x = 0; x < mWidth; ++x)
					for (uint32_t y = 0; y < mHeight; ++y)
						for (uint32_t c = 0; c < mChannels; ++c)
							if (!InputPixel(x, y, c))
								continue;
				break;
			}
			case ShapeOrder::ChannelsHeightWidth:
			{
				for (uint32_t c = 0; c < mChannels; ++c)
					for (uint32_t y = 0; y < mHeight; ++y)
						for (uint32_t x = 0; x < mWidth; ++x)
							if (!InputPixel(x, y, c))
								continue;
				break;
			}
			case ShapeOrder::ChannelsWidthHeight:
			{
				for (uint32_t c = 0; c < mChannels; ++c)
					for (uint32_t x = 0; x < mWidth; ++x)
						for (uint32_t y = 0; y < mHeight; ++y)
							if (!InputPixel(x, y, c))
								continue;
				break;
			}
			default:
				std::cerr << "Unsupported Shape Order." << std::endl;
				return false;
		}

		

		if (input_data.empty())
		{
			std::cerr << "Failed to convert image data to tensor format." << std::endl;
			return false;
		}

		switch (mShapeOrder)
		{
			case ShapeOrder::WidthHeightChannels:
				output = cppflow::tensor(input_data, { 1, static_cast<int64_t>(mWidth), static_cast<int64_t>(mHeight), static_cast<int64_t>(mChannels) });
				break;
			case ShapeOrder::HeightWidthChannels:
				output = cppflow::tensor(input_data, { 1, static_cast<int64_t>(mHeight), static_cast<int64_t>(mWidth), static_cast<int64_t>(mChannels) });
				break;
			case ShapeOrder::ChannelsHeightWidth:
				output = cppflow::tensor(input_data, { 1, static_cast<int64_t>(mChannels), static_cast<int64_t>(mHeight), static_cast<int64_t>(mWidth) });
				break;
			case ShapeOrder::ChannelsWidthHeight:
				output = cppflow::tensor(input_data, { 1, static_cast<int64_t>(mChannels), static_cast<int64_t>(mWidth), static_cast<int64_t>(mHeight) });
				break;
			default:
			{
				std::cerr << "Invalid Shape Order." << std::endl;
				return false;
			}
		}
		return true;
	}
}