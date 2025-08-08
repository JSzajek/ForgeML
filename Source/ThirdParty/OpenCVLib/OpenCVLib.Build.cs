using UnrealBuildTool;
using System.IO;

public class OpenCVLib : ModuleRules
{
	public OpenCVLib(ReadOnlyTargetRules Target) : base(Target)
	{
		Type = ModuleType.External;

		if (Target.Platform == UnrealTargetPlatform.Win64)
		{
            System.Console.WriteLine("Linking OpenCV Library.");

            string ThirdPartyLibsPath = Path.Combine(PluginDirectory, "ThirdParty/");
            string ModulePath = ModuleDirectory + "/";
            
            // Include UE Library Includes Path
            PublicSystemIncludePaths.Add(ModulePath + "Include/");

            // Include TPL includes path
            PublicSystemIncludePaths.Add(ThirdPartyLibsPath + "OpenCV/include/");

            // Add TPL Library and Binaries -------------------------------------------------------
            string LibPath = Path.Combine(ThirdPartyLibsPath, "OpenCV/Release/lib");
            DirectoryInfo libInfo = new DirectoryInfo(LibPath);
            if (libInfo.Exists)
            {
                FileInfo[] files = libInfo.GetFiles("*.lib");
                foreach (FileInfo file in files)
                {
                    PublicAdditionalLibraries.Add(file.FullName);
                }
            }


            DirectoryInfo dllinfo2 = new DirectoryInfo(Path.Combine(ThirdPartyLibsPath, "OpenCV/Release/bin"));
            if (dllinfo2.Exists)
            {
                FileInfo[] files = dllinfo2.GetFiles("*.dll");
                foreach (FileInfo file in files)
                {
                    RuntimeDependencies.Add("$(BinaryOutputDir)/" + file.Name, file.FullName);
                }
            }
            // ------------------------------------------------------------------------------------
        }
    }
}

