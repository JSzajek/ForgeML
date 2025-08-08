using UnrealBuildTool;
using System.IO;

public class CppFlow : ModuleRules
{
	public CppFlow(ReadOnlyTargetRules Target) : base(Target)
	{
		Type = ModuleType.External;

		if (Target.Platform == UnrealTargetPlatform.Win64)
		{
            System.Console.WriteLine("Linking CppFlow Library.");

            string ThirdPartyLibsPath = Path.Combine(PluginDirectory, "ThirdParty/");
            string ModulePath = ModuleDirectory + "/";

            // Include UE Library Includes Path
            PublicSystemIncludePaths.Add(ModulePath + "Include/");
            
            // Include TPL includes path
            PublicSystemIncludePaths.Add(ThirdPartyLibsPath + "cppflow_lib/include/");
            PublicSystemIncludePaths.Add(ThirdPartyLibsPath + "cppflow_lib/libtensorflow-cpu-windows-x86_64/include");

            // Add TPL Library and Binaries -------------------------------------------------------
            PublicAdditionalLibraries.Add(ThirdPartyLibsPath + "cppflow_lib/libtensorflow-cpu-windows-x86_64/lib/tensorflow.lib");

            RuntimeDependencies.Add("$(BinaryOutputDir)/tensorflow.dll", ThirdPartyLibsPath + "cppflow_lib/libtensorflow-cpu-windows-x86_64/lib/tensorflow.dll");
            // ------------------------------------------------------------------------------------
        }
	}
}

