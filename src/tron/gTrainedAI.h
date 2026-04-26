/*

*************************************************************************

ArmageTron -- Just another Tron Lightcycle Game in 3D.
Copyright (C) 2026

**************************************************************************

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

***************************************************************************

*/

#ifndef ArmageTron_TRAINEDAI_H
#define ArmageTron_TRAINEDAI_H

#include "gAIBase.h"

//! menu/config accessors
bool & gTrainedAI_Enable();
bool & gTrainedAI_Learn();
bool & gTrainedAI_Record();
bool & gTrainedAI_Autostart();
int & gTrainedAI_BotCount();
//! this project uses one shared neural AI identity: "Blacklight"
char const * gTrainedAI_Name();

//! Record a classic bot decision as a supervised teacher example.
void gTrainedAI_RecordTeacherDecision( gCycle * cycle, int turn );
//! Close out the current teacher episode for a classic bot.
void gTrainedAI_RecordTeacherEpisodeResult( gCycle * cycle, bool survived, REAL distance );

//! installs the trained AI factory when enabled via configuration
void gTrainedAI_InstallFactoryIfEnabled();

#endif
