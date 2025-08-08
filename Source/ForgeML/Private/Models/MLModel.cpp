#include "Models/MLModel.h"

#include "Utils/ConsoleUtils.h"

#include "Interfaces/IPluginManager.h"
#include "GenericPlatform/GenericPlatformProcess.h"


namespace TF
{
	MLModel::MLModel(const std::string& modelname,
					 const std::filesystem::path& output)
		: mName(modelname),
		mpModel(nullptr)
	{
		mLayout.mModelName = modelname;

		FString TempDir = !output.empty() ? FString(output.c_str()) : FPaths::Combine(FPaths::ProjectSavedDir(), TEXT("Temp/Models"));
		if (!IFileManager::Get().DirectoryExists(*TempDir))
			IFileManager::Get().MakeDirectory(*TempDir, true);

		mOutputDirectory = std::filesystem::canonical(TCHAR_TO_UTF8(*TempDir)).string();

		std::filesystem::path pluginDir = TCHAR_TO_UTF8(*(IPluginManager::Get().FindPlugin("ForgeML")->GetBaseDir()));

		mScriptDirectory = std::filesystem::canonical(pluginDir / "PythonScripts").string();
	}

	bool MLModel::LoadFrom(const std::filesystem::path& loadpath,
						   const std::filesystem::path& output)
	{
		std::string output_path = loadpath.string();
		mName = loadpath.stem().string();

		if (loadpath.has_extension() && loadpath.extension() == ".onnx")
		{
			if (output.empty())
			{
				mOutputDirectory = std::filesystem::canonical(loadpath.parent_path()).string();
			}
			else
			{
				// Ensure output path exists
				if (std::filesystem::is_directory(output))
				{
					if (!std::filesystem::exists(output))
						std::filesystem::create_directory(output);
				}
				else
				{
					std::cerr << "Invalid Output Path: " << output << std::endl;
					return false;
				}

				mOutputDirectory = (std::filesystem::canonical(output)).string();
			}

			output_path = CreateModelName();

			if (!ConvertModelToSavedModel(loadpath.string(), output_path))
				return false;
		}


		const std::string model_path = CreateModelName();
		if (!std::filesystem::exists(loadpath))
		{
			std::cerr << "Load Model Path Does Not Exist: " << loadpath << std::endl;
			return false;
		}


		std::stringstream cmd;
		cmd << "python \"" 
			<< mScriptDirectory 
			<< "/extract_model_info.py\""
			<< " \"" << output_path << "\"";

		if (!ConsoleUtils::Execute(cmd.str().c_str()))
		{
			std::cerr << "Failed to Extract Info From SavedModel {" << output_path << "}" << std::endl;
			return false;
		}

		// Load JSON with input/output tensor names
		std::ifstream in(output_path + "/cppflow_io_names.json");
		if (!in.is_open())
		{
			std::cerr << "Failed to open cppflow_io_names.json" << std::endl;
			return false;
		}

		nlohmann::json io_names;
		in >> io_names;

		for (auto& [key, val] : io_names["outputs"].items())
		{
			const std::string ioName = val.get<std::string>();
			mOutputIONamesMap[ioName] = key;
			mOutputIONames.push_back(ioName);
		}

		for (auto& [key, val] : io_names["inputs"].items())
			mInputToIONamesMap[key] = val.get<std::string>();


		{
			const std::scoped_lock lock(mModelMutex);
			mpModel = std::make_unique<cppflow::model>(model_path);
		}
		return true;
	}

	void MLModel::AddInput(const std::string& name,
						   DataType dtype,
						   std::vector<int> shape, 
						   DomainType domain)
	{
		mLayout.mInputs.push_back(
		{ 
			name, 
			dtype, 
			shape, 
			domain 
		});
	}

	void MLModel::AddOutput(const std::string& name)
	{
		mLayout.mOutputs.push_back(
		{ 
			name 
		});
	}

	void MLModel::AddLayer(LayerType type,
						   const std::unordered_map<std::string, nlohmann::json>& params)
	{
		mLayout.mLayers.push_back(
		{ 
			type, 
			params 
		});
	}

	void MLModel::AddTrainingData(const std::string& input_name, 
								  const nlohmann::json& input_values,
								  const std::string& label_name,
								  const nlohmann::json& label_outputs)
	{
		NamedInput input;
		input.mName = input_name;
		input.mData = input_values;

		NamedLabel label;
		label.mName = label_name;
		label.mData = label_outputs;

		mCurrentTrainingBatch.mInputs.push_back(input);
		mCurrentTrainingBatch.mLabels.push_back(label);
	}

	void MLModel::SaveLayoutJson(const std::filesystem::path& path) const
	{
		mLayout.WriteToFile(path);
	}

	void MLModel::SaveTrainingJson(const std::filesystem::path& path) const
	{
		mCurrentTrainingBatch.WriteToFile(path);
	}

	bool MLModel::CreateModel()
	{
		mOutputIONames.clear();

		const std::string model_path_root = GetModelRoot();

		// Write the layout to a file
		const std::string model_description_path = model_path_root + "/model_description.json";
		mLayout.WriteToFile(model_description_path);

		// Run the Python script to create the model
		std::stringstream python_script;
		python_script << "python \"" 
					  << mScriptDirectory 
					  << "/build_model_from_json.py\"" 
					  << " \"" << model_path_root << "\""
					  << " \"0\"";

		std::string output;
		if (!ConsoleUtils::Execute(python_script.str().c_str(), &output))
		{
			std::cerr << "Failed Model Creation {" << mName << "}: \n\t" << output << std::endl;
			return false;
		}

		mModelVersion = 0;

		UE_LOG(LogTemp, Log, TEXT("%s"), *FString(output.c_str()));

		// Load JSON with input/output tensor names
		const std::string model_path = CreateModelName();
		const std::string io_names_path = model_path + "/cppflow_io_names.json";
		std::ifstream in(io_names_path);
		if (!in.is_open())
		{
			std::cerr << "Failed to open cppflow_io_names.json" << std::endl;
			return false;
		}

		nlohmann::json io_names;
		in >> io_names;

		for (auto& [key, val] : io_names["outputs"].items())
		{
			const std::string ioName = val.get<std::string>();
			mOutputIONamesMap[ioName] = key;
			mOutputIONames.push_back(ioName);
		}

		for (auto& [key, val] : io_names["inputs"].items())
			mInputToIONamesMap[key] = val.get<std::string>();

		{
			const std::scoped_lock lock(mModelMutex);
			mpModel = std::make_unique<cppflow::model>(model_path);
		}

		return true;
	}

	bool MLModel::TrainModel(uint32_t epochs,
							 uint32_t batchSize, 
							 float learning_rate, 
							 bool shuffle, 
							 float validation_split)
	{
		if (mCurrentTrainingBatch.mInputs.empty() || mCurrentTrainingBatch.mLabels.empty())
			return false;


		TF::TrainingConfig config;
		config.epochs			= epochs;
		config.batch_size		= batchSize;
		config.learning_rate	= learning_rate;
		config.shuffle			= shuffle;
		config.validation_split = validation_split;

		const std::string model_path_root = GetModelRoot();
		std::string training_config_path = model_path_root + "/train/train_config.json";
		std::string training_data_path = model_path_root + "/train/train_data.json";

		config.WriteToFile(training_config_path);

		mCurrentTrainingBatch.WriteToFile(model_path_root + "/train/train_data.json");

		std::stringstream trainCmd;
		trainCmd << "python \"" 
				 << mScriptDirectory 
				 << "/train_model_from_json.py\""
				 << " \"" << model_path_root << "\""
				 << " \"" << mModelVersion.load() << "\""
				 << " \"" << mModelVersion.load() + 1 << "\""
				 << " \"" << training_config_path << "\""
				 << " \"" << training_data_path << "\"";

		std::string output;
		if (!ConsoleUtils::Execute(trainCmd.str().c_str(), &output))
		{
			std::cerr << "Failed Execute Training On {" << mName << "}: \n\t" << output << std::endl;
			return false;
		}

		UE_LOG(LogTemp, Log, TEXT("%s"), *FString(output.c_str()));

		// Update Model
		{
			const std::scoped_lock lock(mModelMutex);
			const std::string output_path = CreateModelName(++mModelVersion);
			mpModel = std::make_unique<cppflow::model>(output_path);
		}

		return true;
	}

	bool MLModel::Run(const LabeledTensor& input_tensors,
					  LabeledTensor& output)
	{
		{
			const std::scoped_lock lock(mModelMutex);
			if (!mpModel)
				return false;
		}

		if (mOutputIONamesMap.empty())
			return false;

		std::vector<std::tuple<std::string, cppflow::tensor>> inputs_vec;
		for (const auto& [name, tensor] : input_tensors)
		{
			auto found = mInputToIONamesMap.find(name);
			if (found == mInputToIONamesMap.end())
			{
				std::cerr << "Input name '" << name << "' not found in model input names." << std::endl;
				continue;
			}

			inputs_vec.emplace_back(found->second, tensor);
		}

		std::vector<cppflow::tensor> results;
		{
			const std::scoped_lock lock(mModelMutex);
			results = std::move((*mpModel)(inputs_vec, mOutputIONames));
		}

		for (size_t i = 0; i < results.size(); ++i)
		{
			const std::string& output_name = mOutputIONames[i];
			auto found = mOutputIONamesMap.find(output_name);
			if (found == mOutputIONamesMap.end())
			{
				std::cerr << "Output name '" << output_name << "' not found in model output names." << std::endl;
				continue;
			}
			else
			{
				output[found->second] = results[i];
			}
		}
		return true;
	}

	void MLModel::ExportAll(const std::filesystem::path& directory) const
	{
		std::filesystem::path dir_path(directory);
		if (!std::filesystem::exists(dir_path))
			std::filesystem::create_directories(dir_path);

		mLayout.WriteToFile(dir_path / "model_layout.json");
		mCurrentTrainingBatch.WriteToFile(dir_path / "train_data.json");

		std::cout << "Model and Training Data Exported to: " << dir_path << std::endl;
	}

	bool MLModel::ConvertModelToSavedModel(const std::filesystem::path& filepath,
										   const std::filesystem::path& outputpath)
	{
		std::stringstream cmd;
		cmd << "python \"" 
			<< mScriptDirectory 
			<< "/convert_onnx_to_saved_model.py\""
		    << " \"" << filepath.string() << "\""
		    << " \"" << outputpath.string() << "\"";

		int32_t exit_code = std::system(cmd.str().c_str());
		if (exit_code != 0)
		{
			std::cerr << "Failed to convert model to SavedModel format. Exit code: " << exit_code << std::endl;
			return false;
		}
		return true;	
	}

	std::string MLModel::GetModelRoot() const
	{
		return mOutputDirectory + "/" + mName;
	}

	std::string MLModel::CreateModelName(int32_t version) const
	{
		if(version < 0)
		{
			return mOutputDirectory + "/" + mName + "/Saved_" + std::to_string(mModelVersion.load());
		}
		else
		{
			return mOutputDirectory + "/" + mName + "/Saved_" + std::to_string(version);
		}
	}
}