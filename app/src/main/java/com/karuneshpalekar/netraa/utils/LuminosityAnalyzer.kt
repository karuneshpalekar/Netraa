package com.karuneshpalekar.netraa.utils

import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer
import java.util.ArrayDeque

object LuminosityAnalyzer {

    private const val frameRateWindow = 8
    private val frameTimestamps = ArrayDeque<Long>(5)
    var framesPerSecond: Double = -1.0
        private set
    private val luminosityQueue = LuminosityQueue()


    fun luminosityAnalyze(image: ImageProxy) {
        val lastAnalyzedTimeStamp = calculateTimeFrame()
        val buffer = image.planes[0].buffer
        val data = buffer.toByteArray()
        val pixels = data.map { it.toInt() and 0xFF }
        luminosityQueue.enqueue(pixels.average().toLong())
    }

    fun getLuminosityQueueAverage(): LuminosityState {
        return luminosityQueue.average()
    }

    private fun calculateTimeFrame(): Long {
        val currentTime = System.currentTimeMillis()
        frameTimestamps.push(currentTime)

        while (frameTimestamps.size >= frameRateWindow) frameTimestamps.removeLast()
        val timestampFirst = frameTimestamps.peekFirst() ?: currentTime
        val timestampLast = frameTimestamps.peekLast() ?: currentTime
        framesPerSecond = 1.0 / ((timestampFirst - timestampLast) /
                frameTimestamps.size.coerceAtLeast(1).toDouble()) * 1000.0

        return frameTimestamps.first

    }

    private fun ByteBuffer.toByteArray(): ByteArray {
        rewind()    // Rewind the buffer to zero
        val data = ByteArray(remaining())
        get(data)   // Copy the buffer into a byte array
        return data // Return the byte array
    }


}