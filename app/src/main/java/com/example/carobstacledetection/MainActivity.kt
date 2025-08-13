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
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {
    private lateinit var textViewStatus: TextView
    private lateinit var buttonStartPreview: Button
    private lateinit var buttonStopPreview: Button
    private lateinit var checkBoxProcessing: CheckBox
    private lateinit var imageView: ImageView
    private lateinit var openCvCameraView: CameraBridgeViewBase

    private var isOpenCvInitialized = false
    private var isPreviewActive = false

    private val cameraPermissionRequestCode = 100
    private val targetSize = 600 // Target size 600x600

    private lateinit var inputMat: Mat
    private lateinit var resizedMat: Mat

    companion object {
        private const val TAG = "Camera224"
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
        openCvCameraView = findViewById(R.id.cameraView)

        // Initialize OpenCV
        isOpenCvInitialized = OpenCVLoader.initLocal()

        // Request camera permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                cameraPermissionRequestCode
            )
        }

        openCvCameraView.setCameraIndex(0)
        openCvCameraView.setCvCameraViewListener(this)

        buttonStartPreview.setOnClickListener {
            Log.d(TAG, "Start button clicked")

            // Check camera permission first
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
            } catch (e: Exception) {
                Log.e(TAG, "Error starting camera: ${e.message}")
                textViewStatus.text = "Error: ${e.message}"
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

        var matToDisplay = inputMat

        if (checkBoxProcessing.isChecked) {
            // Convert RGBA to RGB
            val rgbMat = Mat()
            Imgproc.cvtColor(inputMat, rgbMat, Imgproc.COLOR_RGBA2RGB)

            // Resize to 224x224
            Imgproc.resize(rgbMat, resizedMat, Size(targetSize.toDouble(), targetSize.toDouble()))

            // Convert back to RGBA for display
            val displayMat = Mat()
            Imgproc.cvtColor(resizedMat, displayMat, Imgproc.COLOR_RGB2RGBA)

            matToDisplay = displayMat

            // Clean up temporary matrices
            rgbMat.release()
        }

        // Create bitmap for display
        val bitmapToDisplay = Bitmap.createBitmap(
            matToDisplay.cols(),
            matToDisplay.rows(),
            Bitmap.Config.ARGB_8888
        )

        Utils.matToBitmap(matToDisplay, bitmapToDisplay)

        // Display on UI thread
        runOnUiThread {
            imageView.setImageBitmap(bitmapToDisplay)
            if (checkBoxProcessing.isChecked) {
                textViewStatus.text = "Processing: 600x600"
            } else {
                textViewStatus.text = "Original: ${matToDisplay.cols()}x${matToDisplay.rows()}"
            }
        }

        // Clean up display matrix if it was created for processing
        if (checkBoxProcessing.isChecked && matToDisplay != inputMat) {
            matToDisplay.release()
        }

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

        Log.d(TAG, "Activity destroyed")
    }
}