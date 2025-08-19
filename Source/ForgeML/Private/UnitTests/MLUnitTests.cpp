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


			TF::FlatFloatDataBuilder data_builder(3);
			data_builder.AddInputTensor("x", { 3.0f, 7.0f, 1.0f });
			data_builder.AddInputTensor("y", { 4.0f, 2.0f, 8.0f });

			TF::LabeledTensor inputs;
			if (!TestTrue(TEXT("Failed Input Generation!"), data_builder.CreateTensor(inputs)))
				return;

			TF::LabeledTensor output;
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

		It("(3) Train Supervised Model", [this]()
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


			TF::FlatFloatDataBuilder data_builder(4, { 4, 1 });

			data_builder.AddInputTensor("x", { 3.0f, 7.0f, 1.0f, 8.3f });

			TF::LabeledTensor inputs;
			if (!TestTrue(TEXT("Failed Input Generation!"), data_builder.CreateTensor(inputs)))
				return;


			TF::LabeledTensor pre_output;
			bool run_success = model.Run(inputs, pre_output);

			if (!TestTrue(TEXT("Failed To Run Model!"), run_success))
				return;


			// Add Training Data ------------------------------------------------------------------
			model.AddSupervisedTrainingData("x", 
											{ 5.1f, 3.5f, 1.4f, 0.2f }, 
											"y", 
											{ 1.0f, 0.0f });

			model.AddSupervisedTrainingData("x", 
											{ 6.2f, 3.4f, 5.4f, 2.3f }, 
											"y", 
											{ 0, 0.5f });

			model.AddSupervisedTrainingData("x", 
											{ 0.2f, 6.8f, 9.1f, 1.2f }, 
											"y", 
											{ 0, 1.0f });
			// ------------------------------------------------------------------------------------

			bool train_success = model.TrainModel(64);
			if (!TestTrue(TEXT("Failed To Train Model!"), train_success))
				return;


			TF::LabeledTensor post_output;
			run_success = model.Run(inputs, post_output);
			if (!TestTrue(TEXT("Failed To Run Model After Training!"), run_success))
				return;

			cppflow::tensor result = post_output["y"];
			const std::vector<float> post_result_data = result.get_data<float>();
			TestEqual(TEXT("Result Size Mismatch After Training!"), post_result_data.size(), 2u);

			for (size_t i = 0; i < post_result_data.size(); ++i)
			{
				const FString error_str = FString::Printf(TEXT("Result Value Matched After Training! Index: %zu, Pre: %f, Post: %f"),
														  i,
														  pre_output["y"].get_data<float>()[i],
														  post_result_data[i]);

				if (!TestNotEqual(error_str, post_result_data[i], pre_output["y"].get_data<float>()[i]))
					return;
			}
		});

		It("(4) Train Reward Model", [this]()
		{
			TF::MLModel model("predictor");

			model.AddInput("view_state", 
						   TF::DataType::Float32, 
						   { -1, 4, 1 });

			model.AddOutput("action");

			model.AddLayer(TF::LayerType::Flatten,
			{
				{ "input_name", "view_state" },
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
				{ "units", 1 },
				{ "output_name", "action" },
			});


			bool created = model.CreateModel();
			if (!TestTrue(TEXT("Failed To Create Model!"), created))
				return;


			TF::FlatFloatDataBuilder data_builder(4, { 4, 1 });

			data_builder.AddInputTensor("view_state", { 0.0f, 1.0f, 0.0f, 1.0f });

			TF::LabeledTensor inputs;
			if (!TestTrue(TEXT("Failed Input Generation!"), data_builder.CreateTensor(inputs)))
				return;


			TF::LabeledTensor pre_output;
			bool run_success = model.Run(inputs, pre_output);

			if (!TestTrue(TEXT("Failed To Run Model!"), run_success))
				return;


			// Add Training Data ------------------------------------------------------------------
			model.AddRewardData({ 0.1f, 0.5f, 0.3f, 0.0f },
								{ 2.0f },
								1.0f);

			model.AddRewardData({ 0.0f, 0.0f, 0.8f, 0.1f },
								{ 1.0f },
								-0.5f);

			model.AddRewardData({ 0.9f, 0.4f, 0.1f, 0.7f },
								{ 3.0f },
								0.2f);
			// ------------------------------------------------------------------------------------

			bool train_success = model.TrainModel(64);
			if (!TestTrue(TEXT("Failed To Train Model!"), train_success))
				return;


			TF::LabeledTensor post_output;
			run_success = model.Run(inputs, post_output);
			if (!TestTrue(TEXT("Failed To Run Model After Training!"), run_success))
				return;

			cppflow::tensor result = post_output["action"];
			const std::vector<float> post_result_data = result.get_data<float>();
			TestEqual(TEXT("Result Size Mismatch After Training!"), post_result_data.size(), 1u);

			for (size_t i = 0; i < post_result_data.size(); ++i)
			{
				const FString error_str = FString::Printf(TEXT("Result Value Matched After Training! Index: %zu, Pre: %f, Post: %f"),
														  i,
														  pre_output["action"].get_data<float>()[i],
														  post_result_data[i]);

				if (!TestNotEqual(error_str, post_result_data[i], pre_output["action"].get_data<float>()[i]))
					return;
			}
		});

		It("(5) Load Model", [this]()
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

			TF::LabeledTensor results;
			bool runSuccess = model.Run(inputs, results);
			TestTrue(TEXT("Failed To Run Loaded Model!"), runSuccess);
		});

		It("(6) Create & Reload Model", [this]()
		{
			{
				TF::MLModel model("simple_model");

				model.AddInput("x", 
							   TF::DataType::Float32,
							   { -1 });

				model.AddOutput("add_result");

				model.AddLayer(TF::LayerType::Add,
				{
					{ "input_names", { "x" } },
					{ "output_name", "add_result" }
				});

				bool created = model.CreateModel();
				if (!TestTrue(TEXT("Failed To Create Model!"), created))
					return;
			}

			{

				TF::MLModel model2("simple_model");

				bool reloaded = model2.LoadIfExists();

				if (!TestTrue(TEXT("Failed To Reload Model!"), reloaded))
					return;
			}
		});
	});
}