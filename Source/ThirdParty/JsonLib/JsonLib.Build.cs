using UnrealBuildTool;
using System.IO;

public class JsonLib : ModuleRules
{
	public JsonLib(ReadOnlyTargetRules Target) : base(Target)
	{
		Type = ModuleType.External;

        System.Console.WriteLine("Linking Json Library.");

        string ThirdPartyLibsPath = Path.Combine(PluginDirectory, "ThirdParty/");
        string ModulePath = ModuleDirectory + "/";

        // Include UE Library Includes Path
        PublicSystemIncludePaths.Add(ThirdPartyLibsPath + "/json_lib/include/");
        PublicSystemIncludePaths.Add(ModulePath + "Include/");
	}
}

