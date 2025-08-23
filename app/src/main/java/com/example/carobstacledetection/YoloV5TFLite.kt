package com.example.carobstacledetection

import android.content.Context
import android.util.Log
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

class YoloV5TFLite(
    ctx: Context,
    tfliteAssetName: String,
    namesAssetName: String,
    private val inputSize: Int = 320,
    private val confThresh: Float = 0.25f,
    private val iouThresh: Float = 0.45f
) {
    private var interpreter: Interpreter? = null
    private val labels: List<String>
    private val numClasses: Int

    // Input/Output buffers
    private lateinit var inputBuffer: ByteBuffer
    private lateinit var outputBuffer: ByteBuffer

    // Model info
    private var inputShape: IntArray = intArrayOf()
    private var outputShape: IntArray = intArrayOf()
    private var isQuantized = false

    companion object {
        private const val TAG = "YoloV5TFLite"
        private const val BYTES_PER_CHANNEL = 4 // Float32
    }

    init {
        try {
            // Load labels
            val namesPath = AssetUtils.assetToFile(ctx, namesAssetName)
            labels = AssetUtils.readLines(namesPath)
            numClasses = labels.size

            // Load model
            val modelPath = AssetUtils.assetToFile(ctx, tfliteAssetName)
            val model = loadModelFile(modelPath)

            // Create interpreter with GPU delegate if available
            val options = Interpreter.Options()
            val compatList = CompatibilityList()

            if (compatList.isDelegateSupportedOnThisDevice) {
                val delegateOptions = compatList.bestOptionsForThisDevice
                val gpuDelegate = GpuDelegate(delegateOptions)
                options.addDelegate(gpuDelegate)
                Log.d(TAG, "Using GPU delegate for Float16 model")
            } else {
                Log.d(TAG, "GPU delegate not supported, using CPU for Float16 model")
                options.setNumThreads(4)
            }

            interpreter = Interpreter(model, options)

            // Get model info
            val inputTensor = interpreter!!.getInputTensor(0)
            val outputTensor = interpreter!!.getOutputTensor(0)

            inputShape = inputTensor.shape()
            outputShape = outputTensor.shape()
            isQuantized = inputTensor.dataType() == org.tensorflow.lite.DataType.UINT8

            Log.d(TAG, "Float16 YOLOv5su model loaded successfully")
            Log.d(TAG, "Input shape: ${inputShape.contentToString()}")
            Log.d(TAG, "Output shape: ${outputShape.contentToString()}")
            Log.d(TAG, "Input data type: ${inputTensor.dataType()}")
            Log.d(TAG, "Output data type: ${outputTensor.dataType()}")
            Log.d(TAG, "Is quantized: $isQuantized")
            Log.d(TAG, "Number of classes in names file: $numClasses")

            // Allocate buffers
            allocateBuffers()

        } catch (e: Exception) {
            Log.e(TAG, "Error initializing Float16 TFLite model: ${e.message}")
            throw e
        }
    }

    private fun loadModelFile(modelPath: String): MappedByteBuffer {
        val fileInputStream = FileInputStream(modelPath)
        val fileChannel = fileInputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileChannel.size())
    }

    private fun allocateBuffers() {
        // Input buffer: NHWC format for TFLite
        val inputPixels = inputShape[1] * inputShape[2] * inputShape[3]
        inputBuffer = ByteBuffer.allocateDirect(inputPixels * BYTES_PER_CHANNEL)
        inputBuffer.order(ByteOrder.nativeOrder())

        // Output buffer
        val outputSize = outputShape.fold(1) { acc, dim -> acc * dim }
        outputBuffer = ByteBuffer.allocateDirect(outputSize * BYTES_PER_CHANNEL)
        outputBuffer.order(ByteOrder.nativeOrder())

        Log.d(TAG, "Allocated buffers - Input: ${inputBuffer.capacity()}, Output: ${outputBuffer.capacity()}")
    }

    fun detectAndRender(rgbFrame: Mat, rgbaFrame: Mat) {
        try {
            Log.d(TAG, "Starting TFLite detection on frame: ${rgbFrame.width()}x${rgbFrame.height()}")

            val srcW = rgbFrame.width()
            val srcH = rgbFrame.height()

            // Preprocess image
            val scale = min(inputSize.toFloat() / srcW, inputSize.toFloat() / srcH)
            val nw = (srcW * scale).toInt()
            val nh = (srcH * scale).toInt()

            val resized = Mat()
            Imgproc.resize(rgbFrame, resized, Size(nw.toDouble(), nh.toDouble()))

            val padded = Mat.zeros(Size(inputSize.toDouble(), inputSize.toDouble()), resized.type())
            val dx = (inputSize - nw) / 2
            val dy = (inputSize - nh) / 2
            val roi = padded.submat(dy, dy + nh, dx, dx + nw)
            resized.copyTo(roi)

            Log.d(TAG, "Preprocessed image: ${padded.width()}x${padded.height()}")

            // Convert Mat to ByteBuffer (NHWC format)
            convertMatToBuffer(padded, inputBuffer)

            // Run inference
            inputBuffer.rewind()
            outputBuffer.rewind()

            val startTime = System.currentTimeMillis()
            interpreter?.run(inputBuffer, outputBuffer)
            val inferenceTime = System.currentTimeMillis() - startTime

            Log.d(TAG, "Model inference completed in ${inferenceTime}ms")

            // Parse output
            outputBuffer.rewind()
            val detections = parseOutput(outputBuffer, scale, dx, dy, srcW, srcH)

            Log.d(TAG, "Found ${detections.size} raw detections")

            // Apply NMS
            val finalDetections = applyNMS(detections)

            Log.d(TAG, "After NMS: ${finalDetections.size} detections")

            // Draw detections
            drawDetections(rgbaFrame, finalDetections)

            if (finalDetections.isNotEmpty()) {
                Log.i(TAG, "Successfully detected and drew ${finalDetections.size} objects")
            } else {
                Log.w(TAG, "No objects detected in this frame")
            }

            // Cleanup
            resized.release()
            padded.release()
            roi.release()

        } catch (e: Exception) {
            Log.e(TAG, "Detection failed: ${e.message}")
            e.printStackTrace()

            // Draw error message
            Imgproc.putText(rgbaFrame, "TFLite Error: ${e.message}",
                Point(10.0, 30.0), Imgproc.FONT_HERSHEY_SIMPLEX,
                0.6, Scalar(0.0, 0.0, 255.0, 255.0), 2)
        }
    }

    private fun convertMatToBuffer(mat: Mat, buffer: ByteBuffer) {
        buffer.rewind()

        val data = ByteArray((mat.total() * mat.channels()).toInt())
        mat.get(0, 0, data)

        // Convert BGR to RGB and normalize
        // Float16 models may be more sensitive to input range
        for (i in 0 until mat.height()) {
            for (j in 0 until mat.width()) {
                val idx = (i * mat.width() + j).toInt() * 3

                // OpenCV uses BGR, TFLite expects RGB
                val b = data[idx].toInt() and 0xFF
                val g = data[idx + 1].toInt() and 0xFF
                val r = data[idx + 2].toInt() and 0xFF

                if (isQuantized) {
                    buffer.put(r.toByte())
                    buffer.put(g.toByte())
                    buffer.put(b.toByte())
                } else {
                    // For Float16 models, ensure exact normalization
                    val normalizedR = r / 255.0f
                    val normalizedG = g / 255.0f
                    val normalizedB = b / 255.0f

                    buffer.putFloat(normalizedR)
                    buffer.putFloat(normalizedG)
                    buffer.putFloat(normalizedB)
                }
            }
        }

        Log.v(TAG, "Converted Mat to buffer for Float16 model")
    }

    data class Detection(
        val x: Float, val y: Float, val w: Float, val h: Float,
        val confidence: Float, val classId: Int
    )

    private fun parseOutput(buffer: ByteBuffer, scale: Float, dx: Int, dy: Int,
                            srcW: Int, srcH: Int): List<Detection> {
        val detections = mutableListOf<Detection>()
        buffer.rewind()

        // Output format analysis - need to determine based on actual output shape
        // Common formats: [1, N, 85] or [1, 85, N] where 85 = 5 + numClasses
        val stride = 5 + numClasses

        // Output format analysis - YOLOv5su may have different format
        // Shape [1, 8, 2100] suggests: 1 batch, 8 outputs per prediction, 2100 predictions
        // This is different from standard [1, N, 85] format

        Log.d(TAG, "Parsing output with shape: ${outputShape.contentToString()}")

        val numPredictions: Int
        val outputsPerPrediction: Int

        if (outputShape.contentEquals(intArrayOf(1, 8, 2100))) {
            // YOLOv5su format: [1, 8, 2100] = 1 batch, 8 outputs, 2100 predictions
            // 8 outputs = 4 (bbox) + 1 (objectness) + 3 (class scores)
            numPredictions = outputShape[2] // 2100
            outputsPerPrediction = outputShape[1] // 8

            val actualNumClasses = outputsPerPrediction - 5 // 8 - 5 = 3 classes
            Log.w(TAG, "YOLOv5su detected: $actualNumClasses classes in model output")
            Log.w(TAG, "Your names file has $numClasses classes")

            if (actualNumClasses != numClasses) {
                Log.e(TAG, "CLASS MISMATCH! Model has $actualNumClasses classes, names file has $numClasses")
                Log.e(TAG, "This will cause incorrect class predictions!")
            }
        } else {
            // Standard format
            val stride = 5 + numClasses
            numPredictions = when {
                outputShape.size == 3 && outputShape[2] == stride -> outputShape[1] // [1, N, 85]
                outputShape.size == 3 && outputShape[1] == stride -> outputShape[2] // [1, 85, N]
                outputShape.size == 2 && outputShape[1] == stride -> outputShape[0] // [N, 85]
                else -> {
                    Log.w(TAG, "Unknown output format, trying to parse anyway")
                    buffer.remaining() / (stride * 4) // Assume float32
                }
            }
            outputsPerPrediction = stride
        }

        val isTransposed = outputShape.size >= 2 && outputShape[1] == outputsPerPrediction

        Log.d(TAG, "Parsing $numPredictions predictions with $outputsPerPrediction outputs each, transposed=$isTransposed")

        // Debug: Print first few raw values
        val tempBuffer = buffer.duplicate()
        Log.d(TAG, "First 10 raw values:")
        for (i in 0 until minOf(10, tempBuffer.remaining() / 4)) {
            Log.d(TAG, "  [$i] = ${tempBuffer.float}")
        }
        buffer.rewind()

        var detectionsAboveThreshold = 0

        for (i in 0 until numPredictions) {
            val getValue: (Int) -> Float = { c ->
                val index = if (isTransposed) {
                    (c * numPredictions + i) * 4 // Float32 = 4 bytes
                } else {
                    (i * outputsPerPrediction + c) * 4
                }

                if (index + 4 <= buffer.limit()) {
                    buffer.getFloat(index)
                } else {
                    0.0f
                }
            }

            val cx = getValue(0)
            val cy = getValue(1)
            val w = getValue(2)
            val h = getValue(3)
            val objectness = getValue(4)

            // Float16 models may have different objectness ranges
            // Be more strict with objectness filtering
            if (objectness <= 0.05f) continue  // Higher threshold for Float16

            // Handle different output formats
            var bestClass = -1
            var bestScore = 0f

            val numClassOutputs = outputsPerPrediction - 5

            for (c in 5 until outputsPerPrediction) {
                val classScore = getValue(c)

                // For Float16 YOLOv5su, class scores may not need objectness multiplication
                val finalScore = if (outputsPerPrediction == 8) {
                    // YOLOv5su format - class scores are already post-processed
                    classScore
                } else {
                    // Traditional format
                    classScore * objectness
                }

                if (finalScore > bestScore) {
                    bestScore = finalScore
                    bestClass = c - 5
                }
            }

            // Additional Float16-specific filtering
            val boxArea = w * h

            // Filter unrealistic detections common in Float16 models
            if (boxArea < 0.002f || boxArea > 0.9f) continue  // Stricter area constraints
            if (w < 0.02f || h < 0.02f) continue              // Too small
            if (w > 0.95f || h > 0.95f) continue              // Too large

            // Filter detections at exact edges (common Float16 artifacts)
            if (cx <= 0.02f || cx >= 0.98f || cy <= 0.02f || cy >= 0.98f) continue

            // Filter very thin or very wide boxes (often false positives)
            val aspectRatio = w / h
            if (aspectRatio < 0.1f || aspectRatio > 10.0f) continue

            if (bestScore >= confThresh && bestClass >= 0) {
                detectionsAboveThreshold++

                // YOLOv5su outputs normalized coordinates (0.0-1.0)
                // Convert to pixel coordinates first, then map to original image space
                val pixelCx = cx * inputSize
                val pixelCy = cy * inputSize
                val pixelW = w * inputSize
                val pixelH = h * inputSize

                val x1 = ((pixelCx - pixelW/2 - dx) / scale).coerceIn(0f, srcW.toFloat() - 1)
                val y1 = ((pixelCy - pixelH/2 - dy) / scale).coerceIn(0f, srcH.toFloat() - 1)
                val x2 = ((pixelCx + pixelW/2 - dx) / scale).coerceIn(0f, srcW.toFloat() - 1)
                val y2 = ((pixelCy + pixelH/2 - dy) / scale).coerceIn(0f, srcH.toFloat() - 1)

                detections.add(Detection(
                    x1, y1, x2 - x1, y2 - y1,
                    bestScore, bestClass
                ))

                if (detectionsAboveThreshold <= 3) { // Log first few detections
                    val className = if (bestClass < labels.size) labels[bestClass] else "cls$bestClass"
                    Log.d(TAG, "Detection $i: $className, conf=$bestScore")
                    Log.d(TAG, "  Raw coords: cx=$cx, cy=$cy, w=$w, h=$h (normalized)")
                    Log.d(TAG, "  Pixel coords: cx=${cx*inputSize}, cy=${cy*inputSize}, w=${w*inputSize}, h=${h*inputSize}")
                    Log.d(TAG, "  Final box: ($x1,$y1,${x2-x1},${y2-y1}) (original image space)")
                }
            }
        }

        Log.d(TAG, "Found $detectionsAboveThreshold detections above threshold $confThresh")
        return detections
    }

    private fun applyNMS(detections: List<Detection>): List<Detection> {
        if (detections.isEmpty()) return emptyList()

        // Group by class
        val byClass = detections.groupBy { it.classId }
        val result = mutableListOf<Detection>()

        for ((_, classDetections) in byClass) {
            val sorted = classDetections.sortedByDescending { it.confidence }
            val keep = mutableListOf<Detection>()

            for (detection in sorted) {
                var shouldKeep = true

                for (kept in keep) {
                    val iou = calculateIoU(detection, kept)
                    if (iou > iouThresh) {
                        shouldKeep = false
                        break
                    }
                }

                if (shouldKeep) {
                    keep.add(detection)
                }
            }

            result.addAll(keep)
        }

        return result
    }

    private fun calculateIoU(det1: Detection, det2: Detection): Float {
        val x1 = maxOf(det1.x, det2.x)
        val y1 = maxOf(det1.y, det2.y)
        val x2 = minOf(det1.x + det1.w, det2.x + det2.w)
        val y2 = minOf(det1.y + det1.h, det2.y + det2.h)

        val intersection = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
        val area1 = det1.w * det1.h
        val area2 = det2.w * det2.h
        val union = area1 + area2 - intersection

        return if (union > 0) intersection / union else 0f
    }

    private fun drawDetections(frame: Mat, detections: List<Detection>) {
        for (detection in detections) {
            val x1 = detection.x.toDouble()
            val y1 = detection.y.toDouble()
            val x2 = (detection.x + detection.w).toDouble()
            val y2 = (detection.y + detection.h).toDouble()

            // Draw bounding box
            Imgproc.rectangle(frame, Point(x1, y1), Point(x2, y2),
                Scalar(0.0, 255.0, 0.0, 255.0), 3)

            // Draw label
            val className = if (detection.classId < labels.size)
                labels[detection.classId] else "cls${detection.classId}"
            val label = "$className ${String.format("%.2f", detection.confidence)}"

            val textSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, 2, null)
            Imgproc.rectangle(frame,
                Point(x1, y1 - textSize.height - 10),
                Point(x1 + textSize.width, y1),
                Scalar(0.0, 255.0, 0.0, 255.0), -1)

            Imgproc.putText(frame, label, Point(x1, y1 - 5),
                Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255.0, 255.0, 255.0, 255.0), 2)

            Log.d(TAG, "Drew: $label at ($x1,$y1,$x2,$y2)")
        }
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}