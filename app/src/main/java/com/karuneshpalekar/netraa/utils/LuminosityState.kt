package com.karuneshpalekar.netraa.utils

sealed class LuminosityState {

    object LessLight : LuminosityState()

    object AdequateLight : LuminosityState()

    object ExcessLight : LuminosityState()

    object UnknownState : LuminosityState()

}