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

class ImageClassifierHelper(
    var threshold: Float = 0.5f,
    var numThreads: Int = 2,
    var maxResults: Int = 3,
    var currentDelegate: Int = 0,
    var currentModel: Int = 0,
    val context: Context,
    val imageClassifierListener: ClassifierListener
) {
    private var imageClassifierHumanAnimal: ImageClassifier? = null
    private var imageClassifierFurniture: ImageClassifier? = null
    private var imageClassifierBRG: ImageClassifier? = null
    private var imageClassifierAnimals: ImageClassifier? = null

    //Queue
    private var humanAnimalQueue = DetectionQueue(Constants.HUMAN_ANIMAL)
    private var furnitureQueue = DetectionQueue(Constants.FURNITURE)
    private var brgQueue = DetectionQueue(Constants.BRG)
    private var animalQueue = DetectionQueue(Constants.ANIMALS)

    init {
        setupHumanAnimalImageClassifier()
        setupFurnitureImageClassifier()
        setupBRGImageClassifier()
        setupAnimalImageClassifier()
    }

    fun clearImageClassifier() {
        imageClassifierHumanAnimal = null
        imageClassifierFurniture = null
        imageClassifierBRG = null
        imageClassifierAnimals = null
    }

    private fun setupHumanAnimalImageClassifier() {
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
                    //  imageClassifierListener?.onError("GPU is not supported on this device")
                }
            }
            DELEGATE_NNAPI -> {
                baseOptionsBuilder.useNnapi()
            }
        }

        optionsBuilder.setBaseOptions(baseOptionsBuilder.build())

        val modelName = "objecthumananimal_metadata.tflite"

        try {
            imageClassifierHumanAnimal =
                ImageClassifier.createFromFileAndOptions(context, modelName, optionsBuilder.build())

        } catch (e: IllegalStateException) {
            Log.e(TAG, "TFLite failed to load model with error: " + e.message)
        }
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


    private fun setupBRGImageClassifier() {
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

        val modelName = "imageClassification_BRG_metadata.tflite"

        try {
            imageClassifierBRG =
                ImageClassifier.createFromFileAndOptions(context, modelName, optionsBuilder.build())
        } catch (e: IllegalStateException) {
//            imageClassifierListener?.onError(
//                "Image classifier failed to initialize. See error logs for details"
//            )
            Log.e(TAG, "TFLite failed to load model with error: " + e.message)
        }
    }

    private fun setupAnimalImageClassifier() {
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

        val modelName = "imageClassification_animals_metadata.tflite"

        try {
            imageClassifierAnimals =
                ImageClassifier.createFromFileAndOptions(context, modelName, optionsBuilder.build())
        } catch (e: IllegalStateException) {
//            imageClassifierListener?.onError(
//                "Image classifier failed to initialize. See error logs for details"
//            )
            Log.e(TAG, "TFLite failed to load model with error: " + e.message)
        }
    }

    fun classify(image: Bitmap, rotation: Int) {
        if (imageClassifierHumanAnimal == null) {
            setupHumanAnimalImageClassifier()
        }

        if (imageClassifierFurniture == null) {
            setupFurnitureImageClassifier()
        }

        if (imageClassifierBRG == null) {
            setupBRGImageClassifier()
        }

        if (imageClassifierAnimals == null) {
            setupAnimalImageClassifier()
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

        val humanAnimalResults =
            imageClassifierHumanAnimal?.classify(tensorImage, imageProcessingOptions)


        humanAnimalResults.let { it ->
            if (it?.isNotEmpty() == true) {
                it[0].categories.sortedBy {
                    it?.index
                }.forEach {
                    val score = (it.score * 100).toInt()
                    if (humanAnimalQueue.peek() != null) {
                        if (humanAnimalQueue.peek()!!.label == it.label) {
                            humanAnimalQueue.enqueue(Element(it.label, score, tensorImage))
                        } else {
                            humanAnimalQueue.clear()
                            humanAnimalQueue.enqueue(Element(it.label, score, tensorImage))
                        }
                    } else {
                        humanAnimalQueue.enqueue(Element(it.label, score, tensorImage))
                    }

                    if (humanAnimalQueue.count >= 4 &&
                        humanAnimalQueue.average() >= DetectionQueue.HUMAN_ANIMAL_MIN_THRESHOLD
                    ) {

                        if (humanAnimalQueue.peek()?.label == "human") {

                            humanAnimalQueue.peek()?.tensorImage?.let { tI ->
                                imageClassifierListener.onHumanDetection(
                                    tI
                                )
                            }

                            val furnitureResults =
                                imageClassifierFurniture?.classify(
                                    humanAnimalQueue.peek()?.tensorImage,
                                    imageProcessingOptions
                                )


                            val brgResults = imageClassifierBRG?.classify(
                                humanAnimalQueue.peek()?.tensorImage,
                                imageProcessingOptions
                            )

                            if (furnitureResults?.isNotEmpty() == true) {
                                furnitureResults[0].categories.sortedBy { furnitureCategory ->
                                    furnitureCategory?.index
                                }.forEach { fnCategory ->
                                    val furnitureScore = (fnCategory.score * 100).toInt()
                                    Log.d(
                                        "ODH",
                                        "The furniture score is ${fnCategory.label} and the score is " +
                                                "${fnCategory.score}"
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
                                        imageClassifierListener.backgroundResults(
                                            this?.label ?: "Nothing found", this?.score ?: 0
                                        )

                                    }

                                }

                            } else if (brgResults?.isNotEmpty() == true) {

                                brgResults[0].categories.sortedBy { brgCategory ->
                                    brgCategory?.index
                                }.forEach { bCategory ->
                                    val brgScore = (bCategory.score * 100).toInt()

                                    if (brgQueue.peek() != null) {
                                        if (brgQueue.peek()!!.label == bCategory.label) {
                                            brgQueue.enqueue(
                                                Element(
                                                    bCategory.label,
                                                    brgScore,
                                                    tensorImage
                                                )
                                            )
                                        } else {
                                            brgQueue.clear()
                                            brgQueue.enqueue(
                                                Element(
                                                    bCategory.label,
                                                    brgScore,
                                                    tensorImage
                                                )
                                            )
                                        }
                                    } else {
                                        brgQueue.enqueue(
                                            Element(
                                                bCategory.label,
                                                brgScore,
                                                tensorImage
                                            )
                                        )
                                    }
                                }

                                if (brgQueue.count >= 5 &&
                                    brgQueue.average() >= DetectionQueue.BGR_MIN_THRESHOLD
                                ) {
                                    with(brgQueue.peek()) {
                                        imageClassifierListener.backgroundResults(
                                            this?.label ?: "Nothing found", this?.score ?: 0
                                        )

                                    }

                                }
                            }

                        } else {

                            val animalResults = imageClassifierAnimals?.classify(
                                humanAnimalQueue.peek()?.tensorImage,
                                imageProcessingOptions
                            )
                            if (animalResults?.isNotEmpty() == true) {

                                animalResults[0].categories.sortedBy { anCategory ->
                                    anCategory?.index
                                }.forEach { aCategory ->
                                    val anScore = (aCategory.score * 100).toInt()
                                    Log.d(
                                        "Karunesh",
                                        "The animal score is $anScore and the label is ${aCategory.label}"
                                    )
                                    if (animalQueue.peek() != null) {
                                        if (animalQueue.peek()!!.label == aCategory.label) {
                                            animalQueue.enqueue(
                                                Element(
                                                    aCategory.label,
                                                    anScore,
                                                    tensorImage
                                                )
                                            )
                                        } else {
                                            animalQueue.clear()
                                            animalQueue.enqueue(
                                                Element(
                                                    aCategory.label,
                                                    anScore,
                                                    tensorImage
                                                )
                                            )
                                        }
                                    } else {
                                        animalQueue.enqueue(
                                            Element(
                                                aCategory.label,
                                                anScore,
                                                tensorImage
                                            )
                                        )
                                    }
                                }

                                if (animalQueue.count >= 5 &&
                                    animalQueue.average() >= DetectionQueue.BGR_MIN_THRESHOLD
                                ) {
                                    with(animalQueue.peek()) {
                                        imageClassifierListener.onAnimalDetection(
                                            this?.label ?: "Nothing found", this?.score ?: 0
                                        )

                                    }

                                }
                            }

                        }
                    }
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

    interface ClassifierListener {

        fun onHumanDetection(tensorImage: TensorImage)

        fun onAnimalDetection(
            result: String,
            score: Int
        )

        fun backgroundResults(
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
