package com.karuneshpalekar.netraa.utils

import android.content.Context
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.label.Category
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.ObjectDetector

class ObjectDetectorHelper(
    var threshold: Float = 0.5f,
    var numThreads: Int = 2,
    var maxResults: Int = 3,
    var currentDelegate: Int = 0,
    var currentModel: Int = 0,
    val context: Context,
    val detectorListener: DetectorListener
) {

    // For this example this needs to be a var so it can be reset on changes. If the ObjectDetector
    // will not change, a lazy val would be preferable.
    private var objectDetector: ObjectDetector? = null
    private var objectDetector2: ObjectDetector? = null

    private val postureDetectionQueue = DetectionQueue(Constants.POSTURE)
    private val humanClassifierDetectionQueue = DetectionQueue(Constants.HUMAN_CLASSIFIER)

    init {
        setupObjectDetector()
        setupSecondObjectDetector()
    }

    fun clearObjectDetector() {
        objectDetector = null
    }

    // Initialize the object detector using current settings on the
    // thread that is using it. CPU and NNAPI delegates can be used with detectors
    // that are created on the main thread and used on a background thread, but
    // the GPU delegate needs to be used on the thread that initialized the detector
    private fun setupObjectDetector() {
        // Create the base options for the detector using specifies max results and score threshold
        val optionsBuilder =
            ObjectDetector.ObjectDetectorOptions.builder()
                .setScoreThreshold(threshold)
                .setMaxResults(maxResults)

        // Set general detection options, including number of used threads
        val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)

        // Use the specified hardware for running the model. Default to CPU
        when (currentDelegate) {
            DELEGATE_CPU -> {
                // Default
            }
            DELEGATE_GPU -> {
                if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                    baseOptionsBuilder.useGpu()
                } else {
                    //  objectDetectorListener?.onError("GPU is not supported on this device")
                }
            }
            DELEGATE_NNAPI -> {
                baseOptionsBuilder.useNnapi()
            }
        }

        optionsBuilder.setBaseOptions(baseOptionsBuilder.build())

        val modelName = "humanClassifier_metadata.tflite"

        try {
            objectDetector =
                ObjectDetector.createFromFileAndOptions(context, modelName, optionsBuilder.build())
        } catch (e: IllegalStateException) {
//            objectDetectorListener?.onError(
//                "Object detector failed to initialize. See error logs for details"
//            )
            Log.e("Test", "TFLite failed to load model with error: " + e.message)
        }
    }

    private fun setupSecondObjectDetector() {
        // Create the base options for the detector using specifies max results and score threshold
        val optionsBuilder =
            ObjectDetector.ObjectDetectorOptions.builder()
                .setScoreThreshold(threshold)
                .setMaxResults(maxResults)

        // Set general detection options, including number of used threads
        val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)

        // Use the specified hardware for running the model. Default to CPU
        when (currentDelegate) {
            DELEGATE_CPU -> {
                // Default
            }
            DELEGATE_GPU -> {
                if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                    baseOptionsBuilder.useGpu()
                } else {
                    //  objectDetectorListener?.onError("GPU is not supported on this device")
                }
            }
            DELEGATE_NNAPI -> {
                baseOptionsBuilder.useNnapi()
            }
        }

        optionsBuilder.setBaseOptions(baseOptionsBuilder.build())

        val modelName = "posture_model_metadata.tflite"

        try {
            objectDetector2 =
                ObjectDetector.createFromFileAndOptions(context, modelName, optionsBuilder.build())
        } catch (e: IllegalStateException) {
//            objectDetectorListener?.onError(
//                "Object detector failed to initialize. See error logs for details"
//            )
            Log.e("Test", "TFLite failed to load model with error: " + e.message)
        }
    }


    fun detect(tensorImage: TensorImage) {
        Log.d("ODH", "detect is called")
        if (objectDetector == null) {
            setupObjectDetector()
        }

        if (objectDetector2 == null) {
            setupSecondObjectDetector()
        }

        var inferenceTime = SystemClock.uptimeMillis()
        val humanClassifierResults = objectDetector?.detect(tensorImage)
        val secondResults = objectDetector2?.detect(tensorImage)

        if (humanClassifierResults?.isNotEmpty() == true) {

            for (i in humanClassifierResults) {
                for (j in i.categories) {
                    val score = (j.score * 100).toInt()
                    Log.d("ODH", "The score is $score and the label is ${j.label}")
                    if (humanClassifierDetectionQueue.peek() != null) {
                        if (humanClassifierDetectionQueue.peek()!!.label == j.label) {
                            humanClassifierDetectionQueue.enqueue(
                                Element(
                                    j.label,
                                    score,
                                    tensorImage,
                                    j,
                                    i.boundingBox
                                )
                            )
                        } else {
                            humanClassifierDetectionQueue.clear()
                            humanClassifierDetectionQueue.enqueue(
                                Element(
                                    j.label,
                                    score,
                                    tensorImage,
                                    j,
                                    i.boundingBox
                                )
                            )
                        }
                    } else {
                        humanClassifierDetectionQueue.enqueue(
                            Element(
                                j.label, score, tensorImage,
                                j,
                                i.boundingBox
                            )
                        )
                    }
                }

                if (humanClassifierDetectionQueue.count >= 3 &&
                    humanClassifierDetectionQueue.average() >= DetectionQueue.HUMAN_CLASSIFIER_MIN_THRESHOLD
                ) {
                    with(humanClassifierDetectionQueue.peek()) {
                        this?.category?.let { category ->
                            this.boundingBox?.let { bb ->
                                detectorListener.humanClassifierResults(
                                    boundingBox = bb, category = category
                                )
                            }
                        }

                    }

                }

            }
        }

        if (secondResults != null) {
            for (i in secondResults) {
                for (j in i.categories) {
                    val score = (j.score * 100).toInt()
                    if (postureDetectionQueue.peek() != null) {
                        if (postureDetectionQueue.peek()!!.label == j.label) {
                            postureDetectionQueue.enqueue(
                                Element(
                                    j.label,
                                    score,
                                    tensorImage,
                                    j,
                                    i.boundingBox
                                )
                            )
                        } else {
                            postureDetectionQueue.clear()
                            postureDetectionQueue.enqueue(
                                Element(
                                    j.label,
                                    score,
                                    tensorImage,
                                    j,
                                    i.boundingBox
                                )
                            )
                        }
                    } else {
                        postureDetectionQueue.enqueue(
                            Element(
                                j.label, score, tensorImage,
                                j,
                                i.boundingBox
                            )
                        )
                    }
                }


            }
        }

        if (postureDetectionQueue.count >= 3 &&
            postureDetectionQueue.average() >= DetectionQueue.POSTURE_CLASSIFIER_MIN_THRESHOLD
        ) {
            with(postureDetectionQueue.peek()) {
                this?.category?.let { category ->
                    this.boundingBox?.let { bb ->
                        detectorListener.postureDetectionResults(
                            boundingBox = bb, category = category
                        )
                    }
                }

            }

        }
    }


    interface DetectorListener {

        fun humanClassifierResults(
            boundingBox: RectF,
            category: Category
        )

        fun postureDetectionResults(
            boundingBox: RectF,
            category: Category
        )

    }

    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
        const val MODEL_MOBILENETV1 = 0
        const val MODEL_EFFICIENTDETV0 = 1
        const val MODEL_EFFICIENTDETV1 = 2
        const val MODEL_EFFICIENTDETV2 = 3
    }
}
