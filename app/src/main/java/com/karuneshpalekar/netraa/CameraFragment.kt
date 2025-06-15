package com.karuneshpalekar.netraa

import android.animation.Animator
import android.annotation.SuppressLint
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Bundle
import android.os.CountDownTimer
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.util.DisplayMetrics
import android.util.Log
import android.view.*
import android.widget.Toast
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import androidx.navigation.fragment.findNavController
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.karuneshpalekar.netraa.databinding.FragmentFirstBinding
import com.karuneshpalekar.netraa.utils.ImageClassifierBackgroundHelper
import com.karuneshpalekar.netraa.utils.ImageClassifierHelper
import com.karuneshpalekar.netraa.utils.LuminosityAnalyzer.getLuminosityQueueAverage
import com.karuneshpalekar.netraa.utils.LuminosityAnalyzer.luminosityAnalyze
import com.karuneshpalekar.netraa.utils.LuminosityState.*
import com.karuneshpalekar.netraa.utils.ObjectDetectorHelper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.label.Category
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


class CameraFragment : Fragment(), ImageClassifierHelper.ClassifierListener,
    ObjectDetectorHelper.DetectorListener,
    ImageClassifierBackgroundHelper.BackgroundClassifierListener {


    //Bool
    private var backgroundBool: Boolean? = false
    private var genderBool: Boolean? = false
    private var postureBool: Boolean? = false
    private var animalBool: Boolean? = false
    private var humanBool: Boolean? = false
    private var focusBool: Boolean? = true

    //Previous
    private var prevBackgroundResult: String? = null
    private var prevAnimal: String? = null
    private var prevHumanClassifier: HumanClassifier? = null
    private var prevPosture: Posture? = null

    //Current
    private var backgroundResult: String? = null
    private var animal: String? = null
    private var humanClassifier: HumanClassifier? = null
    private var posture: Posture? = null

    //CountDown Timer
    private lateinit var countDownAnimalTimer: CountDownTimer
    private lateinit var countDownHumanTimer: CountDownTimer
    private lateinit var countDownFocusTimer: CountDownTimer
    private lateinit var countDownTorchTimer: CountDownTimer

    private var flashMode = ImageCapture.FLASH_MODE_ON
    private var imageCapture: ImageCapture? = null

    val speechRecognizerIntent =
        Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)

    private lateinit var textToSpeech: TextToSpeech
    private lateinit var broadcastManager: LocalBroadcastManager

    private var _fragmentCameraBinding: FragmentFirstBinding? = null
    private val fragmentCameraBinding
        get() = _fragmentCameraBinding!!

    private lateinit var imageClassifierHelper: ImageClassifierHelper
    private lateinit var objectDetectorHelper: ObjectDetectorHelper
    private lateinit var imageClassifierBackground: ImageClassifierBackgroundHelper

    private lateinit var bitmapBuffer: Bitmap
    private lateinit var speechRecognizer: SpeechRecognizer

    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null


    /** Blocking camera operations are performed using this executor */
    private lateinit var cameraExecutor: ExecutorService

    private val volumeDownReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            when (intent.getIntExtra(KEY_EVENT_EXTRA, KeyEvent.KEYCODE_UNKNOWN)) {
                // When the volume down button is pressed, simulate a shutter button click
                KeyEvent.KEYCODE_VOLUME_DOWN -> {
                    lifecycleScope.launch(Dispatchers.Main) {
                        Toast.makeText(context, "Key Down", Toast.LENGTH_SHORT).show()
                        speechRecognizer.startListening(speechRecognizerIntent)
                    }
                }
            }
        }
    }


    override fun onResume() {
        super.onResume()

        if (!PermissionsFragment.hasPermissions(requireContext())) {
            findNavController().navigate(CameraFragmentDirections.actionFirstFragmentToPermissionsFragment())
        }
    }

    override fun onDestroyView() {
        _fragmentCameraBinding = null
        super.onDestroyView()

        // Shut down our background executor
        cameraExecutor.shutdown()
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding = FragmentFirstBinding.inflate(inflater, container, false)

        return fragmentCameraBinding.root
    }

    @SuppressLint("MissingPermission")
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        imageClassifierHelper =
            ImageClassifierHelper(context = requireContext(), imageClassifierListener = this)

        imageClassifierBackground =
            ImageClassifierBackgroundHelper(
                context = requireContext(),
                imageClassifierListener = this
            )

        objectDetectorHelper = ObjectDetectorHelper(
            context = requireContext(), detectorListener = this
        )

        cameraExecutor = Executors.newSingleThreadExecutor()

        fragmentCameraBinding.viewFinder.post {
            // Set up the camera and its use cases
            setUpCamera()
        }


        broadcastManager = LocalBroadcastManager.getInstance(view.context)
        val filter = IntentFilter().apply { addAction(KEY_EVENT_ACTION) }
        broadcastManager.registerReceiver(volumeDownReceiver, filter)

        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(context)

        speechRecognizerIntent.putExtra(
            RecognizerIntent.EXTRA_LANGUAGE_MODEL,
            RecognizerIntent.LANGUAGE_MODEL_FREE_FORM
        )
        speechRecognizerIntent.putExtra(
            RecognizerIntent.EXTRA_LANGUAGE,
            Locale.getDefault()
        )

        speechRecognizer.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {
                textToSpeech.speak(
                    "Listening",
                    TextToSpeech.QUEUE_FLUSH,
                    null
                )
            }

            override fun onBeginningOfSpeech() {

            }

            override fun onRmsChanged(rmsdB: Float) {
            }

            override fun onBufferReceived(buffer: ByteArray?) {
            }

            override fun onEndOfSpeech() {
                textToSpeech.speak(
                    "Trying to Summarize what you said",
                    TextToSpeech.QUEUE_FLUSH,
                    null
                )
            }

            override fun onError(error: Int) {
                textToSpeech.speak(
                    "Did not get you. Please Try Again !!",
                    TextToSpeech.QUEUE_FLUSH,
                    null
                )
            }

            override fun onResults(results: Bundle?) {
                val data =
                    results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                //  binding.speechToText.text = data?.get(0) ?: getString(R.string.error)

                val input = data?.get(0)
                textToSpeech.speak(
                    "Searching for $input",
                    TextToSpeech.QUEUE_FLUSH,
                    null
                )

                val words = input?.split(" ")
                if (words != null) {
                    for (word in words) {

                        if (word.lowercase() == "read") {
                            recognizeText(InputImage.fromBitmap(bitmapBuffer, 0))
                        }

                        if (word.lowercase() == "background"||word.lowercase() == "sit"||word.lowercase() == "seat") {
                            imageClassifierBackground.classify(bitmapBuffer,getScreenOrientation())
                        }
                    }
                }


            }

            override fun onPartialResults(partialResults: Bundle?) {
            }

            override fun onEvent(eventType: Int, params: Bundle?) {
            }

        })

        textToSpeech = TextToSpeech(context) { p0 ->
            if (p0 != TextToSpeech.ERROR) {
                textToSpeech.language = Locale.UK
            }
        }


//        backgroundBool = true
//        genderBool = true
//        postureBool = true
//        animalBool = true
        // Attach listeners to UI control widgets
        // initBottomSheetControls()
    }

    // Initialize CameraX, and prepare to bind the camera use cases
    private fun setUpCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener(
            {
                // CameraProvider
                cameraProvider = cameraProviderFuture.get()

                // Build and bind the camera use cases
                bindCameraUseCases()
            },
            ContextCompat.getMainExecutor(requireContext())
        )
    }


    // Update the values displayed in the bottom sheet. Reset classifier.
    private fun updateControlsUi() {

        // Needs to be cleared instead of reinitialized because the GPU
        // delegate needs to be initialized on the thread using it when applicable
        imageClassifierHelper.clearImageClassifier()
        imageClassifierBackground.clearImageClassifier()
        objectDetectorHelper.clearObjectDetector()
        fragmentCameraBinding.overlay.clear()
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        imageAnalyzer?.targetRotation = fragmentCameraBinding.viewFinder.display.rotation
    }

    // Declare and bind preview, capture and analysis use cases
    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() {

        // CameraProvider
        val cameraProvider =
            cameraProvider ?: throw IllegalStateException("Camera initialization failed.")

        // CameraSelector - makes assumption that we're only using the back camera
        val cameraSelector =
            CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build()

        // Preview. Only using the 4:3 ratio because this is the closest to our models
        preview =
            Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
                .build()

        imageCapture = ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setFlashMode(flashMode)
            .build()

        // ImageAnalysis. Using RGBA 8888 to match how our models work
        imageAnalyzer =
            ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { image ->
                        if (!::bitmapBuffer.isInitialized) {
                            bitmapBuffer = Bitmap.createBitmap(
                                image.width,
                                image.height,
                                Bitmap.Config.ARGB_8888
                            )
                        }
                        classifyImage(image)
                    }
                }


        // Must unbind the use-cases before rebinding them
        cameraProvider.unbindAll()

        try {
            camera = cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageCapture,
                imageAnalyzer
            )
            preview?.setSurfaceProvider(fragmentCameraBinding.viewFinder.surfaceProvider)

        } catch (exc: Exception) {
            //Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun getScreenOrientation(): Int {
        val outMetrics = DisplayMetrics()

        val display: Display?
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
            display = requireActivity().display
            display?.getRealMetrics(outMetrics)
        } else {
            @Suppress("DEPRECATION")
            display = requireActivity().windowManager.defaultDisplay
            @Suppress("DEPRECATION")
            display.getMetrics(outMetrics)
        }

        return display?.rotation ?: 0
    }

    private fun classifyImage(image: ImageProxy) {


        image.use {
            bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer)
            luminosityAnalyze(image)
        }
        val imageRotation = image.imageInfo.rotationDegrees
        if (camera?.cameraInfo?.hasFlashUnit() == true) {
            getLuminosityQueueAverage().let { state ->
                Log.d("State", "The state is $state")
                when (state) {
                    LessLight -> {
                        lifecycleScope.launch {
                            countDownTorchTimerFunc()
                        }
                    }
                    AdequateLight -> {
                        // camera!!.cameraControl.enableTorch(false)
                    }
                    ExcessLight -> {
                        //camera!!.cameraControl.enableTorch(false)
                    }
                    UnknownState -> {
                        //   camera!!.cameraControl.enableTorch(false)
                    }
                    else -> {
                        // camera!!.cameraControl.enableTorch(false)
                    }
                }
            }
            // imageCapture?.flashMode = flashMode
        }

        lifecycleScope.launch {
            if (focusBool == true) {
                countDownFocusTimerFunc()
                tapToFocus()
            }
        }
        imageClassifierHelper.classify(bitmapBuffer, getScreenOrientation())
    }


    private fun recognizeText(image: InputImage) {

        val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

        recognizer.process(image)
            .addOnSuccessListener { visionText ->

                for (block in visionText.textBlocks) {
                    val text = block.text

                    //   Toast.makeText(context,text,Toast.LENGTH_SHORT).show()
                    textToSpeech.speak(
                        "$text ",
                        TextToSpeech.QUEUE_FLUSH, null
                    )
                }

            }
            .addOnFailureListener { e ->
                textToSpeech.speak(
                    "Nothing found to be read",
                    TextToSpeech.QUEUE_FLUSH, null
                )
            }
    }

    private fun countDownAnimalTimerFunc() {
        countDownAnimalTimer = object : CountDownTimer(TWENTY, COUNT_DOWN_INTERVAL) {
            override fun onTick(millisUntilFinished: Long) {
                animalBool = true
            }

            override fun onFinish() {
                animalBool = false
                prevAnimal = null
                Log.d("Timer", "Timer has ended and bool is $animalBool")
            }
        }
        countDownAnimalTimer.start()
    }


    private fun countDownFocusTimerFunc() {
        countDownFocusTimer = object : CountDownTimer(FIVE, COUNT_DOWN_INTERVAL) {
            override fun onTick(millisUntilFinished: Long) {
                focusBool = false
            }

            override fun onFinish() {
                focusBool = true
            }
        }
        countDownFocusTimer.start()
    }

    private fun countDownTorchTimerFunc() {
        countDownTorchTimer = object : CountDownTimer(MIN, COUNT_DOWN_INTERVAL) {
            override fun onTick(millisUntilFinished: Long) {
                camera!!.cameraControl.enableTorch(true)
            }

            override fun onFinish() {
                camera!!.cameraControl.enableTorch(false)
            }
        }
        countDownTorchTimer.start()
    }


    //Human vs Animal results
    override fun onHumanDetection(tensorImage: TensorImage) {
        objectDetectorHelper.detect(tensorImage)
        lifecycleScope.launch(Dispatchers.Main) {
            Log.d("ODH", " HUman bool is true")
            if (humanBool == false) {
                Log.d("ODH", " Human bool is false")
                humanBool = true
                textToSpeech.speak(
                    "There is a human Nearby. Summary will be provided in short ",
                    TextToSpeech.QUEUE_FLUSH, null
                )
                countDownHumanTimerFunc()
            }
        }
    }


    //butterfly, cat, chicken, cow, dog , elephant, horse, sheep ,spider, squirrel
    override fun onAnimalDetection(result: String, score: Int) {
        lifecycleScope.launch(Dispatchers.Main) {
            if (prevAnimal == null) {
                if (animalBool == false) {
                    if (::countDownAnimalTimer.isInitialized) {
                        animalBool = false
                        countDownAnimalTimer.cancel()
                    }
                    prevAnimal = if (result == "elephant") {
                        textToSpeech.speak(
                            "There is an $result Nearby",
                            TextToSpeech.QUEUE_FLUSH, null
                        )
                        result
                    } else {
                        textToSpeech.speak(
                            "There is a $result Nearby",
                            TextToSpeech.QUEUE_FLUSH, null
                        )
                        result
                    }
                    countDownAnimalTimerFunc()
                }
            }


        }
    }

    private fun countDownHumanTimerFunc() {
        countDownHumanTimer = object : CountDownTimer(TWENTY, COUNT_DOWN_INTERVAL) {
            override fun onTick(millisUntilFinished: Long) {
                Log.d("ODH", " $millisUntilFinished")
            }

            override fun onFinish() {
                if (humanClassifier?.label != null && posture?.label != null && backgroundResult != null) {
                    textToSpeech.speak(
                        "There is a ${humanClassifier?.label} ${posture?.label} " +
                                "on a $backgroundResult", TextToSpeech.QUEUE_FLUSH, null
                    )
                } else if (humanClassifier?.label != null) {
                    textToSpeech.speak(
                        "There is a ${humanClassifier?.label} nearby",
                        TextToSpeech.QUEUE_FLUSH,
                        null
                    )
                } else if (backgroundResult != null) {
                    textToSpeech.speak(
                        "There is a $backgroundResult nearby",
                        TextToSpeech.QUEUE_FLUSH,
                        null
                    )
                } else if (humanClassifier?.label != null && posture?.label != null) {
                    textToSpeech.speak(
                        "There is a ${humanClassifier?.label}, ${posture?.label} nearby",
                        TextToSpeech.QUEUE_FLUSH,
                        null
                    )
                } else {
                    textToSpeech.speak(
                        "Unfortunately no Information was found. Trying Again !! ",
                        TextToSpeech.QUEUE_FLUSH,
                        null
                    )
                }


                if (posture?.boundingBox != null) {
                    val area = posture?.boundingBox!!.width() * posture?.boundingBox!!.height()
                    Log.d("Area", "The area of the object is at $area")
                    if (area >= 100000) {
                        textToSpeech.speak(
                            "The object is nearby, ${posture?.label} ",
                            TextToSpeech.QUEUE_FLUSH,
                            null
                        )
                    } else {
                        textToSpeech.speak(
                            "The object is faraway, ${posture?.label} ",
                            TextToSpeech.QUEUE_FLUSH,
                            null
                        )
                    }
                }


                lifecycleScope.launch(Dispatchers.Main) {
                    delay(TEN)
                    genderBool = false
                    humanBool = false
                    backgroundBool = false
                    postureBool = false
                }

            }
        }
        countDownHumanTimer.start()
    }


    //Background detection : Furniture / BRG / Animals
    override fun backgroundResults(result: String, score: Int) {
        lifecycleScope.launch(Dispatchers.Main) {
            if (backgroundBool == false) {
                backgroundBool = true
                backgroundResult = result

                Log.d("ODH", "The background is $backgroundResult")
            }
            //  textToSpeech.speak("The background is a $result", TextToSpeech.QUEUE_FLUSH, null)
        }
    }

    //In-depth human classifier
    override fun humanClassifierResults(boundingBox: RectF, category: Category) {
        lifecycleScope.launch(Dispatchers.Main) {
            Log.d("BoundBox", "The left is ${boundingBox.left}")
            if (genderBool == false) {
                Log.d("BoundBox", "The label is ${category.label}")
                humanClassifier = HumanClassifier(category.label, boundingBox)
                genderBool = true
                if (backgroundResult != null && posture != null) {
                    textToSpeech.speak(
                        "There is a ${humanClassifier?.label} ${posture?.label} " +
                                "on a $backgroundResult", TextToSpeech.QUEUE_FLUSH, null
                    )
                    prevHumanClassifier = humanClassifier
                    prevPosture = posture
                    prevBackgroundResult = backgroundResult

                }
            }
        }
        Log.d(
            "ODH", "The toast says " +
                    "label is ${category.label} and the index is ${category.index}"
        )
    }

    //Posture of humans being detected
    override fun postureDetectionResults(boundingBox: RectF, category: Category) {

        lifecycleScope.launch(Dispatchers.Main) {
            if (postureBool == false) {
                postureBool = true
                posture = Posture(category.label, boundingBox)
            }
        }

    }

    private fun tapToFocus() {

        val midX =
            (fragmentCameraBinding.viewFinder.left + fragmentCameraBinding.viewFinder.right) / 2
        val midY =
            (fragmentCameraBinding.viewFinder.top + fragmentCameraBinding.viewFinder.bottom) / 2

        lifecycleScope.launch(Dispatchers.Main) {
            val meteringPoint = DisplayOrientedMeteringPointFactory(
                fragmentCameraBinding.viewFinder.display,
                camera?.cameraInfo!!,
                fragmentCameraBinding.viewFinder.width.toFloat(),
                fragmentCameraBinding.viewFinder.height.toFloat()
            ).createPoint(midX.toFloat(), midY.toFloat())
            val action = FocusMeteringAction.Builder(meteringPoint).build()
            camera?.cameraControl!!.startFocusAndMetering(action)
            val width: Float = fragmentCameraBinding.focusRing.width.toFloat()
            val height: Float = fragmentCameraBinding.focusRing.height.toFloat()
            fragmentCameraBinding.focusRing.x = midX - width / 2
            fragmentCameraBinding.focusRing.y = midY - height / 2

            fragmentCameraBinding.focusRing.visibility = View.VISIBLE
            fragmentCameraBinding.focusRing.alpha = 1F

            fragmentCameraBinding.focusRing.animate()
                .setStartDelay(400)
                .setDuration(600)
                .alpha(0F)
                .setListener(object : Animator.AnimatorListener {
                    override fun onAnimationStart(p0: Animator) {

                    }

                    override fun onAnimationEnd(p0: Animator) {
                    }

                    override fun onAnimationCancel(p0: Animator) {
                    }

                    override fun onAnimationRepeat(p0: Animator) {
                    }

                })

        }
    }

    override fun onDemandBackgroundResults(result: String, score: Int) {
        textToSpeech.speak(
            "There is a $result to sit",
            TextToSpeech.QUEUE_FLUSH,
            null
        )

    }


    companion object {
        private const val TAG = "Image Classifier"
        const val COUNT_DOWN_INTERVAL = 1000L

        //Timer
        const val ONE = 1000L
        const val TWO = 2000L
        const val THREE = 3000L
        const val FOUR = 4000L
        const val FIVE = 5000L
        const val TEN = 10000L
        const val TWENTY = 20000L
        const val FORTY = 40000L
        const val MIN = 60000L
    }


}

//sitting, standing
data class Posture(
    var label: String?,
    var boundingBox: RectF?
)

data class HumanClassifier(
    var label: String?,
    var boundingBox: RectF?
)
