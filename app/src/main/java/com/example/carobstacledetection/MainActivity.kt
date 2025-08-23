package com.example.carobstacledetection

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.CheckBox
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.JavaCameraView
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {
    private lateinit var textViewStatus: TextView
    private lateinit var buttonStartPreview: Button
    private lateinit var buttonStopPreview: Button
    private lateinit var checkBoxProcessing: CheckBox
    private lateinit var imageView: ImageView
    private lateinit var imageViewProcessed: ImageView
    private lateinit var openCvCameraView: JavaCameraView

    private var isOpenCvInitialized = false
    private var isPreviewActive = false

    private val cameraPermissionRequestCode = 100
    private val targetSize = 320
    private var detector: YoloV5Onnx? = null
    private var detectorTFLite: YoloV5TFLite? = null

    private lateinit var inputMat: Mat
    private lateinit var resizedMat: Mat

    companion object {
        private const val TAG = "Camera320"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        textViewStatus = findViewById(R.id.textViewStatus)
        buttonStartPreview = findViewById(R.id.buttonStartPreview)
        buttonStopPreview = findViewById(R.id.buttonStopPreview)
        checkBoxProcessing = findViewById(R.id.checkboxEnableProcessing)
        imageView = findViewById(R.id.imageView)
        imageViewProcessed = findViewById(R.id.imageViewProcessed)
        openCvCameraView = findViewById(R.id.cameraView)

        // Initialize OpenCV
        if (!OpenCVLoader.initLocal()) {
            Log.e(TAG, "OpenCV initialization failed")
            isOpenCvInitialized = false
        } else {
            Log.d(TAG, "OpenCV initialization succeeded")
            isOpenCvInitialized = true
        }

        // Request camera permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                cameraPermissionRequestCode
            )
        }

        // Setup camera
        openCvCameraView.visibility = CameraBridgeViewBase.VISIBLE
        openCvCameraView.setCvCameraViewListener(this)

        buttonStartPreview.setOnClickListener {
            Log.d(TAG, "Start button clicked")

            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
                Log.w(TAG, "Camera permission not granted, requesting...")
                ActivityCompat.requestPermissions(
                    this,
                    arrayOf(Manifest.permission.CAMERA),
                    cameraPermissionRequestCode
                )
                return@setOnClickListener
            }

            try {
                Log.d(TAG, "Attempting to start camera...")
                openCvCameraView.setCameraPermissionGranted()
                openCvCameraView.enableView()
                Log.d(TAG, "Camera enableView() called")

//                detector = YoloV5Onnx(this@MainActivity,
//                    "best.onnx", "coco.names",
//                    inputSize = 640, confThresh = 0.35f, iouThresh = 0.45f)

//                detector = YoloV5Onnx(this@MainActivity,
//                    "yolov5n-320.onnx", "coco.names",
//                    inputSize = 320, confThresh = 0.1f, iouThresh = 0.3f)  // Match your targetSize

//                detector = YoloV5Onnx(this@MainActivity,
//                    "yolov5s.onnx", "coco.names",
//                    inputSize = 320,  // Match your export size
//                    confThresh = 0.1f,  // Lower threshold for testing
//                    iouThresh = 0.3f)   // Lower IoU threshold

                detectorTFLite = YoloV5TFLite(this@MainActivity,
                    "suBest2_float16.tflite", "custom.names",
                    inputSize = 320,  // Match your model's input size
                    confThresh = 0.1f,  // Higher threshold for TFLite
                    iouThresh = 0.45f)

                Log.d(TAG, "TFLite detector initialized")

                Log.d(TAG, "Detector initialized with low thresholds for testing")

            } catch (e: Exception) {
                Log.e(TAG, "Error initializing detector: ${e.message}")
                detector = null
                textViewStatus.text = "Model loading failed: ${e.message}"
            }

            updateControls()
        }

        buttonStopPreview.setOnClickListener {
            Log.d(TAG, "Stop button clicked")
            try {
                openCvCameraView.disableView()
                Log.d(TAG, "Camera disableView() called")
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping camera: ${e.message}")
            }
            updateControls()
        }

        updateControls()
    }

    private fun updateControls() {
        when {
            !isOpenCvInitialized -> {
                textViewStatus.text = "OpenCV initialization error"
                buttonStartPreview.isEnabled = false
                buttonStopPreview.isEnabled = false
                Log.e(TAG, "OpenCV not initialized")
            }
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                    != PackageManager.PERMISSION_GRANTED -> {
                textViewStatus.text = "Camera permission required"
                buttonStartPreview.isEnabled = true
                buttonStopPreview.isEnabled = false
                Log.w(TAG, "Camera permission not granted")
            }
            else -> {
                textViewStatus.text = if (isPreviewActive) "Camera Active" else "Camera Ready"
                buttonStartPreview.isEnabled = !isPreviewActive
                buttonStopPreview.isEnabled = isPreviewActive
                Log.d(TAG, "Controls updated - Preview active: $isPreviewActive")
            }
        }
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        Log.d(TAG, "onCameraViewStarted called: ${width}x${height}")
        isPreviewActive = true

        inputMat = Mat(height, width, CvType.CV_8UC4)
        resizedMat = Mat(targetSize, targetSize, CvType.CV_8UC3)

        runOnUiThread {
            updateControls()
        }
    }

    override fun onCameraViewStopped() {
        Log.d(TAG, "onCameraViewStopped called")
        isPreviewActive = false

        if (::inputMat.isInitialized) inputMat.release()
        if (::resizedMat.isInitialized) resizedMat.release()

        runOnUiThread {
            updateControls()
        }
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame?): Mat {
        inputFrame!!.rgba().copyTo(inputMat)

        // Convert RGBA to RGB and resize to 320x320
        val rgbMat = Mat()
        Imgproc.cvtColor(inputMat, rgbMat, Imgproc.COLOR_RGBA2RGB)
        Imgproc.resize(rgbMat, resizedMat, Size(targetSize.toDouble(), targetSize.toDouble()))

        // Create original display version for top window
        val originalDisplayMat = Mat()
        Imgproc.cvtColor(resizedMat, originalDisplayMat, Imgproc.COLOR_RGB2RGBA)

        val rgb = Mat()
        Imgproc.cvtColor(originalDisplayMat, rgb, Imgproc.COLOR_RGBA2RGB)

        // Test if drawing works at all
//        Imgproc.rectangle(originalDisplayMat,
//            Point(10.0, 10.0), Point(100.0, 50.0),
//            Scalar(255.0, 0.0, 0.0, 255.0), 3)
//        Imgproc.putText(originalDisplayMat, "TEST DRAW",
//            Point(10.0, 80.0), Imgproc.FONT_HERSHEY_SIMPLEX,
//            1.0, Scalar(255.0, 0.0, 0.0, 255.0), 2)
//
//        Log.d(TAG, "Added test rectangle and text")

//        detector?.detectAndRender(rgb, originalDisplayMat)
        detectorTFLite?.detectAndRender(rgb, originalDisplayMat)

        // Create bitmap for top window (always original)
        val originalBitmap = Bitmap.createBitmap(
            originalDisplayMat.cols(),
            originalDisplayMat.rows(),
            Bitmap.Config.ARGB_8888
        )
        Utils.matToBitmap(originalDisplayMat, originalBitmap)

        // Create processed version for bottom window
        val processedBitmap: Bitmap
        val statusText: String

        if (checkBoxProcessing.isChecked) {
            // Create grayscale version
            val grayscaleMat = Mat()
            Imgproc.cvtColor(resizedMat, grayscaleMat, Imgproc.COLOR_RGB2GRAY)

            val grayscaleDisplayMat = Mat()
            Imgproc.cvtColor(grayscaleMat, grayscaleDisplayMat, Imgproc.COLOR_GRAY2RGBA)

            processedBitmap = Bitmap.createBitmap(
                grayscaleDisplayMat.cols(),
                grayscaleDisplayMat.rows(),
                Bitmap.Config.ARGB_8888
            )
            Utils.matToBitmap(grayscaleDisplayMat, processedBitmap)

            statusText = "Dual View: Original + Grayscale (320x320)"

            // Clean up
            grayscaleMat.release()
            grayscaleDisplayMat.release()
        } else {
            // Use same as original
//            processedBitmap = originalBitmap.copy(Bitmap.Config.ARGB_8888, false)
            statusText = "Dual View: Original + Original (320x320)"
        }

        // Display on UI thread
        runOnUiThread {
            imageView.setImageBitmap(originalBitmap)
//            imageViewProcessed.setImageBitmap(processedBitmap)
            textViewStatus.text = statusText
        }

        // Clean up
        rgbMat.release()
        originalDisplayMat.release()

        return inputMat
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == cameraPermissionRequestCode) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Log.d(TAG, "Camera permission granted")
                updateControls()
            } else {
                Log.e(TAG, "Camera permission denied")
                textViewStatus.text = "Camera permission required"
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()

        if (::inputMat.isInitialized) inputMat.release()
        if (::resizedMat.isInitialized) resizedMat.release()

        detectorTFLite?.close()

        Log.d(TAG, "Activity destroyed")
    }

    override fun onPause() {
        super.onPause()
        if (::openCvCameraView.isInitialized) {
            openCvCameraView.disableView()
        }
    }

    override fun onResume() {
        super.onResume()
        if (isOpenCvInitialized && ::openCvCameraView.isInitialized) {
            openCvCameraView.enableView()
        }
    }
}