// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "Modules/ModuleManager.h"

class FForgeMLModule : public IModuleInterface
{
public:
	/// IModuleInterface Implementation -----------------------------------------------------------

	/// <summary>
	/// Called right after the module DLL has been loaded and the module object has been created
	/// Load dependent modules here, and they will be guaranteed to be available during ShutdownModule.ie:
	///
	/// FModuleManager::Get().LoadModuleChecked(TEXT("HTTP"));
	/// </summary>
	virtual void StartupModule() override;

	/// <summary>
	/// Called before the module is unloaded, right before the module object is destroyed.
	/// During normal shutdown, this is called in reverse order that modules finish StartupModule().
	/// This means that, as long as a module references dependent modules in it's StartupModule(), it
	/// can safely reference those dependencies in ShutdownModule() as well.
	/// </summary>
	virtual void ShutdownModule() override;

	/// -------------------------------------------------------------------------------------------
};
