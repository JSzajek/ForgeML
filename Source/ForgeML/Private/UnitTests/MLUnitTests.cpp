#include "Misc/AutomationTest.h"

#include "Interfaces/IPluginManager.h"

#include "TFModelLib.h"

#include "Kismet/KismetRenderingLibrary.h"

// Reference: https://minifloppy.it/posts/2024/automated-testing-specs-ue5/#writing-tests

BEGIN_DEFINE_SPEC(FMLUnitTestsSpecs, 
				  "ForgeML Unit Test",
				  EAutomationTestFlags::EditorContext |
				  EAutomationTestFlags::CommandletContext |
				  EAutomationTestFlags::ProductFilter);

// Variables and functions defined here will end up being member of
// the FMLUnitTestsSpecs class and will be accessible in the tests

FString ModuleDirectory = IPluginManager::Get().FindPlugin("ForgeML")->GetBaseDir();

FTimespan TestTimeout_S = FTimespan(0, 0, 20); /* 20 Seconds */

END_DEFINE_SPEC(FMLUnitTestsSpecs);

void FMLUnitTestsSpecs::Define()
{
	// Essentials Of The ForgeML (e.g.).
	Describe("Essentials", [this]()
	{
		It("(1) Create Model", [this]()
		{
			TF::MLModel model("simple_add");

			model.AddInput("x", 
						   TF::DataType::Float32,
						   { -1 });

			model.AddInput("y", 
						   TF::DataType::Float32,
						   { -1 });

			model.AddOutput("add_result");

			model.AddLayer(TF::LayerType::Add,
			{
				{ "input_names", { "x", "y" } },
				{ "output_name", "add_result" }
			});

			bool created = model.CreateModel();
			TestTrue(TEXT("Failed To Create Model!"), created);
		});

		It("(2) Run Model", [this]()
		{
			TF::MLModel model("simple_add");

			model.AddInput("x", 
						   TF::DataType::Float32,
						   { -1 });

			model.AddInput("y", 
						   TF::DataType::Float32,
						   { -1 });

			model.AddOutput("add_result");

			model.AddLayer(TF::LayerType::Add,
			{
				{ "input_names", { "x", "y" } },
				{ "output_name", "add_result" }
			});


			bool created = model.CreateModel();
			if (!TestTrue(TEXT("Failed To Create Model!"), created))
				return;


			TF::MLModel::LabeledTensor output;
			std::unordered_map<std::string, cppflow::tensor> inputs;

			inputs["x"] = cppflow::tensor({ 3.0f, 7.0f, 1.0f });
			inputs["y"] = cppflow::tensor({ 4.0f, 2.0f, 8.0f });

			bool run_success = model.Run(inputs, output);

			if (!TestTrue(TEXT("Failed To Run Model!"), run_success))
				return;

			cppflow::tensor result = output["add_result"];
			const std::vector<float> result_data = result.get_data<float>();

			TestEqual(TEXT("Result Size Mismatch!"), result_data.size(), 3u);

			for (size_t i = 0; i < result_data.size(); ++i)
			{
				const FString error_str = FString::Printf(TEXT("Result Value Mismatch! Index: %zu, Expected: %f, Got: %f"),
														  i,
														  inputs["x"].get_data<float>()[i] + inputs["y"].get_data<float>()[i],
														  result_data[i]);

				if (!TestEqual(error_str, result_data[i], inputs["x"].get_data<float>()[i] + inputs["y"].get_data<float>()[i]))
					return;
			}
		});

		It("(3) Train Model", [this]()
		{
			TF::MLModel model("linear");

			model.AddInput("x", 
						   TF::DataType::Float32, 
						   { -1, 4, 1 });

			model.AddOutput("y");

			model.AddLayer(TF::LayerType::Flatten,
			{
				{ "input_name", "x" },
				{ "output_name", "flat_input" }
			});

			model.AddLayer(TF::LayerType::Dense,
			{
				{ "input_name", "flat_input" },
				{ "units", 16 },
				{ "output_name", "dense_1" },
			});

			model.AddLayer(TF::LayerType::Dense,
			{
				{ "input_name", "dense_1" },
				{ "units", 2 },
				{ "output_name", "y" },
			});


			bool created = model.CreateModel();
			if (!TestTrue(TEXT("Failed To Create Model!"), created))
				return;


			TF::MLModel::LabeledTensor pre_output;
			std::unordered_map<std::string, cppflow::tensor> inputs;

			inputs["x"] = cppflow::tensor(std::vector<float>{ 3.0f, 7.0f, 1.0f, 8.3f }, { 4, 1 });

			bool run_success = model.Run(inputs, pre_output);

			if (!TestTrue(TEXT("Failed To Run Model!"), run_success))
				return;


			// Add Training Data ------------------------------------------------------------------
			model.AddTrainingData("x", 
								  { 5.1f, 3.5f, 1.4f, 0.2f }, 
								  "y", 
								  { 1.0f, 0.0f });

			model.AddTrainingData("x", 
								  { 6.2f, 3.4f, 5.4f, 2.3f }, 
								  "y", 
								  { 0, 0.5f });

			model.AddTrainingData("x", 
								  { 0.2f, 6.8f, 9.1f, 1.2f }, 
								  "y", 
								  { 0, 1.0f });
			// ------------------------------------------------------------------------------------

			bool train_success = model.TrainModel(64);
			if (!TestTrue(TEXT("Failed To Train Model!"), train_success))
				return;


			TF::MLModel::LabeledTensor post_output;
			run_success = model.Run(inputs, post_output);
			if (!TestTrue(TEXT("Failed To Run Model After Training!"), run_success))
				return;

			cppflow::tensor result = post_output["y"];
			const std::vector<float> post_result_data = result.get_data<float>();
			TestEqual(TEXT("Result Size Mismatch After Training!"), post_result_data.size(), 2u);

			for (size_t i = 0; i < post_result_data.size(); ++i)
			{
				const FString error_str = FString::Printf(TEXT("Result Value Mismatch After Training! Index: %zu, Expected: %f, Got: %f"),
														  i,
														  pre_output["y"].get_data<float>()[i],
														  post_result_data[i]);

				if (!TestNotEqual(error_str, post_result_data[i], pre_output["y"].get_data<float>()[i]))
					return;
			}
		});

		It("(4) Load Model", [this]()
		{
			TF::MLModel model("BirdClassifier");

			FString modelPath = FPaths::Combine(ModuleDirectory, TEXT("UnitTest/Models/bird-classifier/BirdClassifier.onnx"));
			FString tempOutputDir = FPaths::Combine(FPaths::ProjectSavedDir(), TEXT("Temp/Models"));
			model.LoadFrom(TCHAR_TO_UTF8(*modelPath),
						   TCHAR_TO_UTF8(*tempOutputDir));

			int32_t target_width = 260;
			int32_t target_height = 260;

			TF::ImageTensorLoader image_loader(target_width,
											   target_height,
											   3,
											   true,
											   TF::ChannelOrder::RGB,
											   TF::ShapeOrder::ChannelsHeightWidth);

			FString dataPath = FPaths::Combine(ModuleDirectory, TEXT("UnitTest/Data/"));
			std::string dataPath_str = TCHAR_TO_UTF8(*dataPath);

			// Load Labels
			std::ifstream in(dataPath_str + "/bird_label_map.json");
			if (!TestTrue(TEXT("Failed To Open Label Map!"), in.is_open()))
				return;

			std::unordered_map<std::string, cppflow::tensor> inputs;
			std::string imgPath_str = dataPath_str + "/test_bird_dataset/31/ANNAS HUMMINGBIRD.jpg";
			
			if (!TestTrue(TEXT("Image Path Exists!"), std::filesystem::exists(imgPath_str)))
				return;

			bool loadedImg = image_loader.Load(imgPath_str, inputs["pixel_values"]);
			if (!TestTrue(TEXT("Failed To Load Image!"), loadedImg))
				return;

			TF::MLModel::LabeledTensor results;
			bool runSuccess = model.Run(inputs, results);
			TestTrue(TEXT("Failed To Run Loaded Model!"), runSuccess);
		});
	});
}