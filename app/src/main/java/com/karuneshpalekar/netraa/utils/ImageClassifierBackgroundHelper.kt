package com.karuneshpalekar.netraa.utils

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import android.view.Surface
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.core.vision.ImageProcessingOptions
import org.tensorflow.lite.task.vision.classifier.ImageClassifier

class ImageClassifierBackgroundHelper(
    var threshold: Float = 0.5f,
    var numThreads: Int = 2,
    var maxResults: Int = 3,
    var currentDelegate: Int = 0,
    var currentModel: Int = 0,
    val context: Context,
    val imageClassifierListener: BackgroundClassifierListener
) {
    private var imageClassifierFurniture: ImageClassifier? = null

    //Queue
    private var furnitureQueue = DetectionQueue(Constants.FURNITURE)

    init {
        setupFurnitureImageClassifier()
    }

    fun clearImageClassifier() {
        imageClassifierFurniture = null
    }


    private fun setupFurnitureImageClassifier() {
        val optionsBuilder = ImageClassifier.ImageClassifierOptions.builder()
            .setScoreThreshold(threshold)
            .setMaxResults(maxResults)

        val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)

        when (currentDelegate) {
            DELEGATE_CPU -> {
                // Default
            }
            DELEGATE_GPU -> {
                if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                    baseOptionsBuilder.useGpu()
                } else {
                    //imageClassifierListener?.onError("GPU is not supported on this device")
                }
            }
            DELEGATE_NNAPI -> {
                baseOptionsBuilder.useNnapi()
            }
        }

        optionsBuilder.setBaseOptions(baseOptionsBuilder.build())

        val modelName = "furniture_model_metadata.tflite"

        try {
            imageClassifierFurniture =
                ImageClassifier.createFromFileAndOptions(context, modelName, optionsBuilder.build())
        } catch (e: IllegalStateException) {
//            imageClassifierListener?.onError(
//                "Image classifier failed to initialize. See error logs for details"
//            )
            Log.e(TAG, "TFLite failed to load model with error: " + e.message)
        }
    }


    fun classify(image: Bitmap, rotation: Int) {


        if (imageClassifierFurniture == null) {
            setupFurnitureImageClassifier()
        }

        // Inference time is the difference between the system time at the start and finish of the
        // process
        var inferenceTime = SystemClock.uptimeMillis()

        // Create preprocessor for the image.
        // See https://www.tensorflow.org/lite/inference_with_metadata/
        //            lite_support#imageprocessor_architecture
        val imageProcessor =
            ImageProcessor.Builder()
                .build()

        // Preprocess the image and convert it into a TensorImage for classification.
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))

        val imageProcessingOptions = ImageProcessingOptions.builder()
            .setOrientation(getOrientationFromRotation(rotation))
            .build()
//
//        val humanAnimalResults =
//            imageClassifierHumanAnimal?.classify(tensorImage, imageProcessingOptions)

        val furnitureResults =
            imageClassifierFurniture?.classify(
                tensorImage,
                imageProcessingOptions
            )

        if (furnitureResults?.isNotEmpty() == true) {
            furnitureResults[0].categories.sortedBy { furnitureCategory ->
                furnitureCategory?.index
            }.forEach { fnCategory ->
                val furnitureScore = (fnCategory.score * 100).toInt()
                Log.d(
                    "BackgroundODH",
                    "The furniture score is ${fnCategory.label} and the score is " +
                            "${fnCategory.score}"
                )

                imageClassifierListener.onDemandBackgroundResults(
                    fnCategory.label ?: "Nothing found", fnCategory.score.toInt() ?: 0
                )
                if (furnitureQueue.peek() != null) {
                    if (furnitureQueue.peek()!!.label == fnCategory.label) {
                        furnitureQueue.enqueue(
                            Element(
                                fnCategory.label,
                                furnitureScore,
                                tensorImage
                            )
                        )
                    } else {
                        furnitureQueue.clear()
                        furnitureQueue.enqueue(
                            Element(
                                fnCategory.label,
                                furnitureScore,
                                tensorImage
                            )
                        )
                    }
                } else {
                    furnitureQueue.enqueue(
                        Element(
                            fnCategory.label,
                            furnitureScore,
                            tensorImage
                        )
                    )
                }
            }

            if (furnitureQueue.count >= 6 &&
                furnitureQueue.average() >= DetectionQueue.FURNITURE_MIN_THRESHOLD
            ) {
                with(furnitureQueue.peek()) {


                }

            }


        }

        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

    }


    private fun getOrientationFromRotation(rotation: Int): ImageProcessingOptions.Orientation {
        when (rotation) {
            Surface.ROTATION_270 ->
                return ImageProcessingOptions.Orientation.BOTTOM_RIGHT
            Surface.ROTATION_180 ->
                return ImageProcessingOptions.Orientation.RIGHT_BOTTOM
            Surface.ROTATION_90 ->
                return ImageProcessingOptions.Orientation.TOP_LEFT
            else ->
                return ImageProcessingOptions.Orientation.RIGHT_TOP
        }
    }

    interface BackgroundClassifierListener {


        fun onDemandBackgroundResults(
            result: String,
            score: Int
        )
    }


    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
        const val MODEL_MOBILENETV1 = 0
        const val MODEL_EFFICIENTNETV0 = 1
        const val MODEL_EFFICIENTNETV1 = 2
        const val MODEL_EFFICIENTNETV2 = 3

        private const val TAG = "ImageClassifierHelper"
    }
}
