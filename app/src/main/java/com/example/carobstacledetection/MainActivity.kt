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
import com.example.carobstacledetection.R
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc
import java.io.IOException

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
    private val inputSize = 384 // Target size untuk YOLO input

    private lateinit var inputMat: Mat
    private lateinit var processedMat: Mat
    private lateinit var resizedMat: Mat
    private lateinit var yoloNet: Net
    private var isYoloLoaded = false

    // YOLO parameters
    private val confidenceThreshold = 0.5f
    private val nmsThreshold = 0.4f

    companion object {
        private const val TAG = "YOLOCamera"
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

        isOpenCvInitialized = OpenCVLoader.initLocal()

        // Load YOLO model
        loadYoloModel()

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
            openCvCameraView.setCameraPermissionGranted()
            openCvCameraView.enableView()
            updateControls()
        }

        buttonStopPreview.setOnClickListener {
            openCvCameraView.disableView()
            updateControls()
        }

        updateControls()
    }

    private fun loadYoloModel() {
        try {
            // Pastikan file yolo.onnx ada di folder assets
            val modelInputStream = assets.open("yolo.onnx")
            val modelDir = filesDir
            val modelFile = java.io.File(modelDir, "yolo.onnx")

            modelFile.outputStream().use { output ->
                modelInputStream.copyTo(output)
            }

            yoloNet = Dnn.readNetFromONNX(modelFile.absolutePath)
            isYoloLoaded = !yoloNet.empty()

            if (isYoloLoaded) {
                Log.d(TAG, "YOLO model loaded successfully")
            } else {
                Log.e(TAG, "Failed to load YOLO model")
            }

        } catch (e: IOException) {
            Log.e(TAG, "Error loading YOLO model: ${e.message}")
            isYoloLoaded = false
        }
    }

    private fun updateControls() {
        when {
            !isOpenCvInitialized -> {
                textViewStatus.text = "OpenCV initialization error"
                buttonStartPreview.isEnabled = false
                buttonStopPreview.isEnabled = false
            }
            !isYoloLoaded -> {
                textViewStatus.text = "YOLO model not loaded"
                buttonStartPreview.isEnabled = false
                buttonStopPreview.isEnabled = false
            }
            else -> {
                textViewStatus.text = "OpenCV & YOLO ready"
                buttonStartPreview.isEnabled = !isPreviewActive
                buttonStopPreview.isEnabled = isPreviewActive
            }
        }
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        isPreviewActive = true

        inputMat = Mat(height, width, CvType.CV_8UC4)
        processedMat = Mat(height, width, CvType.CV_8UC3)
        resizedMat = Mat(inputSize, inputSize, CvType.CV_8UC3)

        updateControls()
    }

    override fun onCameraViewStopped() {
        isPreviewActive = false

        if (::inputMat.isInitialized) inputMat.release()
        if (::processedMat.isInitialized) processedMat.release()
        if (::resizedMat.isInitialized) resizedMat.release()

        updateControls()
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame?): Mat {
        inputFrame!!.rgba().copyTo(inputMat)

        var matToDisplay = inputMat

        if (checkBoxProcessing.isChecked && isYoloLoaded) {
            // Convert RGBA to RGB for YOLO
            Imgproc.cvtColor(inputMat, processedMat, Imgproc.COLOR_RGBA2RGB)

            // Resize/crop to 384x384
            val resizedFrame = preprocessForYolo(processedMat)

            // Run YOLO inference
            val detections = runYoloInference(resizedFrame)

            // Draw detections on original frame
            val resultFrame = drawDetections(processedMat, detections)

            matToDisplay = resultFrame
        }

        // Prepare bitmap for display
        val bitmapToDisplay = Bitmap.createBitmap(
            matToDisplay.cols(),
            matToDisplay.rows(),
            Bitmap.Config.ARGB_8888
        )

        // Convert Mat to Bitmap based on format
        if (matToDisplay.channels() == 3) {
            // RGB format, need to convert to RGBA for bitmap
            val rgbaMat = Mat()
            Imgproc.cvtColor(matToDisplay, rgbaMat, Imgproc.COLOR_RGB2RGBA)
            Utils.matToBitmap(rgbaMat, bitmapToDisplay)
            rgbaMat.release()
        } else {
            Utils.matToBitmap(matToDisplay, bitmapToDisplay)
        }

        // Display on UI thread
        runOnUiThread {
            imageView.setImageBitmap(bitmapToDisplay)
        }

        return inputMat
    }

    private fun preprocessForYolo(inputMat: Mat): Mat {
        // Option 1: Resize with interpolation (may distort aspect ratio)
        Imgproc.resize(inputMat, resizedMat, Size(inputSize.toDouble(), inputSize.toDouble()))

        // Option 2: Center crop (preserves aspect ratio)
        // Uncomment below and comment above resize if you prefer center crop
        /*
        val centerCrop = centerCropToSquare(inputMat)
        Imgproc.resize(centerCrop, resizedMat, Size(inputSize.toDouble(), inputSize.toDouble()))
        centerCrop.release()
        */

        return resizedMat
    }

    private fun centerCropToSquare(inputMat: Mat): Mat {
        val height = inputMat.rows()
        val width = inputMat.cols()
        val size = minOf(height, width)

        val startX = (width - size) / 2
        val startY = (height - size) / 2

        val roi = Rect(startX, startY, size, size)
        return Mat(inputMat, roi)
    }

    private fun runYoloInference(inputMat: Mat): List<Detection> {
        val detections = mutableListOf<Detection>()

        try {
            // Create blob from image
            val blob = Dnn.blobFromImage(
                inputMat,
                1.0 / 255.0,  // Scale factor
                Size(inputSize.toDouble(), inputSize.toDouble()),
                Scalar(0.0),  // Mean subtraction
                true,         // Swap RB
                false,        // Crop
                CvType.CV_32F
            )

            // Set input to network
            yoloNet.setInput(blob)

            // Run forward pass
            val outputs = mutableListOf<Mat>()
            yoloNet.forward(outputs, yoloNet.unconnectedOutLayersNames)

            // Process outputs
            for (output in outputs) {
                val rows = output.size(1).toInt()
                val cols = output.size(2).toInt()

                for (i in 0 until rows) {
                    val data = FloatArray(cols)
                    output.get(intArrayOf(0, i), data)

                    // Extract detection data
                    val confidence = data[4]
                    if (confidence > confidenceThreshold) {
                        val centerX = data[0] * inputSize
                        val centerY = data[1] * inputSize
                        val width = data[2] * inputSize
                        val height = data[3] * inputSize

                        val left = centerX - width / 2
                        val top = centerY - height / 2

                        // Find class with highest score
                        var maxClassScore = 0f
                        var classId = 0
                        for (j in 5 until cols) {
                            if (data[j] > maxClassScore) {
                                maxClassScore = data[j]
                                classId = j - 5
                            }
                        }

                        val finalConfidence = confidence * maxClassScore
                        if (finalConfidence > confidenceThreshold) {
                            detections.add(
                                Detection(
                                Rect2d(left.toDouble(), top.toDouble(), width.toDouble(), height.toDouble()),
                                finalConfidence,
                                classId
                            )
                            )
                        }
                    }
                }
            }

            // Clean up
            outputs.forEach { it.release() }
            blob.release()

        } catch (e: Exception) {
            Log.e(TAG, "YOLO inference error: ${e.message}")
        }

        return detections
    }

    private fun drawDetections(inputMat: Mat, detections: List<Detection>): Mat {
        val resultMat = inputMat.clone()

        for (detection in detections) {
            val rect = detection.box
            val scaleX = inputMat.cols().toDouble() / inputSize
            val scaleY = inputMat.rows().toDouble() / inputSize

            // Scale coordinates back to original image size
            val scaledRect = Rect(
                (rect.x * scaleX).toInt(),
                (rect.y * scaleY).toInt(),
                (rect.width * scaleX).toInt(),
                (rect.height * scaleY).toInt()
            )

            // Draw bounding box
            Imgproc.rectangle(
                resultMat,
                Point(scaledRect.x.toDouble(), scaledRect.y.toDouble()),
                Point((scaledRect.x + scaledRect.width).toDouble(),
                    (scaledRect.y + scaledRect.height).toDouble()),
                Scalar(0.0, 255.0, 0.0), // Green color
                3
            )

            // Draw class label and confidence
            val label = "Class ${detection.classId}: ${String.format("%.2f", detection.confidence)}"
            Imgproc.putText(
                resultMat,
                label,
                Point(scaledRect.x.toDouble(), scaledRect.y - 10.0),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                0.6,
                Scalar(0.0, 255.0, 0.0),
                2
            )
        }

        return resultMat
    }

    data class Detection(
        val box: Rect2d,
        val confidence: Float,
        val classId: Int
    )

    override fun onDestroy() {
        super.onDestroy()
        // Tidak perlu release yoloNet karena class Net tidak punya method release()
        // dan akan dibersihkan otomatis oleh garbage collector
    }
}