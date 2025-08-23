package com.example.carobstacledetection

import android.content.Context
import android.util.Log
import org.opencv.core.*
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min

class YoloV5Onnx(
    ctx: Context,
    onnxAssetName: String,
    namesAssetName: String,
    private val inputSize: Int = 320,  // Changed to match your export
    private val confThresh: Float = 0.35f,
    private val iouThresh: Float = 0.45f
) {

    private val net: Net
    private val labels: List<String>
    private val numClasses: Int

    companion object {
        private const val TAG = "YoloV5Onnx"
    }

    init {
        val onnxPath = AssetUtils.assetToFile(ctx, onnxAssetName)
        val namesPath = AssetUtils.assetToFile(ctx, namesAssetName)
        net = Dnn.readNetFromONNX(onnxPath)
        labels = AssetUtils.readLines(namesPath)
        numClasses = labels.size

        Log.d(TAG, "Model loaded successfully")
        Log.d(TAG, "Input size: $inputSize")
        Log.d(TAG, "Number of classes: $numClasses")
        Log.d(TAG, "Confidence threshold: $confThresh")
    }

    // rgbFrame: RGB Mat, rgbaFrame: RGBA Mat for drawing (display)
    fun detectAndRender(rgbFrame: Mat, rgbaFrame: Mat){
        try {
            Log.d(TAG, "Starting detection on frame: ${rgbFrame.width()}x${rgbFrame.height()}")

            // sizes
            val srcW = rgbFrame.width()
            val srcH = rgbFrame.height()

            // letterbox: scale keeping aspect, pad to square inputSize
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

            // blob: model expects RGB normalized to [0,1]
//            val blob = Dnn.blobFromImage(padded, 1.0 / 255.0, Size(inputSize.toDouble(), inputSize.toDouble()),
//                Scalar(0.0, 0.0, 0.0), /*swapRB*/ false, /*crop*/ false)

            val blob = Dnn.blobFromImage(padded, 1.0/255.0, Size(inputSize.toDouble(), inputSize.toDouble()),
                Scalar(0.0, 0.0, 0.0), false, false)

            net.setInput(blob)
            val output = net.forward() // This should work now

//            Log.d(TAG, "Model inference completed")
//            Log.d(TAG, "Output shape: [${output.size(0)}, ${output.size(1)}, ${output.size(2)}]")
//
//// read output to float array
//            val outShape = intArrayOf(output.size(0).toInt(), output.size(1).toInt(), output.size(2).toInt())
//            val data = FloatArray((output.total() * output.channels()).toInt())
//            output[0, 0, data]
//
//// DEBUG: Print first 10 predictions' raw values
//            Log.d(TAG, "=== RAW OUTPUT DEBUG ===")
//            val stride = 5 + numClasses
//            val numPred = if (outShape[2] == stride) outShape[1] else outShape[2]
//            val transposed = outShape[1] == stride
//
//            Log.d(TAG, "Stride: $stride, NumPred: $numPred, Transposed: $transposed")
//
//// Check first 3 predictions in detail
//            for (i in 0 until minOf(3, numPred)) {
//                val getValue: (Int) -> Float = { c ->
//                    if (transposed) {
//                        data[c * numPred + i]
//                    } else {
//                        data[i * stride + c]
//                    }
//                }
//
//                val cx = getValue(0)
//                val cy = getValue(1)
//                val w = getValue(2)
//                val h = getValue(3)
//                val obj = getValue(4)
//
//                // Get top 3 class scores
//                val classScores = mutableListOf<Pair<Int, Float>>()
//                for (c in 5 until stride) {
//                    val classScore = getValue(c)
//                    classScores.add(Pair(c-5, classScore))
//                }
//                classScores.sortByDescending { it.second }
//
//                Log.d(TAG, "Pred $i: cx=$cx, cy=$cy, w=$w, h=$h, obj=$obj")
//                Log.d(TAG, "  Top classes: ${classScores.take(3)}")
//                Log.d(TAG, "  Max class*obj: ${classScores[0].second * obj}")
//            }
//
//// Check if any objectness scores are reasonable
//            var maxObj = 0f
//            var maxClassScore = 0f
//            var maxFinalScore = 0f
//            for (i in 0 until numPred) {
//                val getValue: (Int) -> Float = { c ->
//                    if (transposed) data[c * numPred + i] else data[i * stride + c]
//                }
//                val obj = getValue(4)
//                maxObj = maxOf(maxObj, obj)
//
//                for (c in 5 until stride) {
//                    val classScore = getValue(c)
//                    maxClassScore = maxOf(maxClassScore, classScore)
//                    maxFinalScore = maxOf(maxFinalScore, classScore * obj)
//                }
//            }
//
//            Log.d(TAG, "Max objectness: $maxObj")
//            Log.d(TAG, "Max class score: $maxClassScore")
//            Log.d(TAG, "Max final score: $maxFinalScore")
//            Log.d(TAG, "=== END RAW OUTPUT DEBUG ===")

// Continue with existing detection logic...

            Log.d(TAG, "Model inference completed")
            Log.d(TAG, "Output shape: [${output.size(0)}, ${output.size(1)}, ${output.size(2)}]")

            // read output to float array
            val outShape = intArrayOf(output.size(0).toInt(), output.size(1).toInt(), output.size(2).toInt())
            // flatten into float[]
            val data = FloatArray((output.total() * output.channels()).toInt())
            output[0, 0, data] // read entire mat into data (works with CV_32F mats)

            // determine layout: stride = 5 + numClasses
            val stride = 5 + numClasses
            val numPred: Int
            val transposed: Boolean
            if (outShape[1] == stride) {
                // [1, 85, N] -> transpose case
                numPred = outShape[2]
                transposed = true
            } else if (outShape[2] == stride) {
                // [1, N, 85]
                numPred = outShape[1]
                transposed = false
            } else {
                // fallback: try to deduce
                numPred = data.size / stride
                transposed = false
            }

            Log.d(TAG, "Processing $numPred predictions, stride=$stride, transposed=$transposed")

            val boxes = mutableListOf<Rect2d>()
            val scores = mutableListOf<Float>()
            val classes = mutableListOf<Int>()

            var detectionsAboveThreshold = 0

            for (i in 0 until numPred) {
                // compute base index depending on layout
                val getValue: (Int) -> Float = { c ->
                    if (transposed) {
                        data[c * numPred + i]
                    } else {
                        data[i * stride + c]
                    }
                }

                val obj = getValue(4)
                if (obj <= 1e-6) continue

                // find best class and score
                var bestCls = -1
                var bestConf = 0f
                for (c in 5 until stride) {
                    val conf = getValue(c) * obj
                    if (conf > bestConf) {
                        bestConf = conf
                        bestCls = c - 5
                    }
                }

                if (bestConf >= confThresh && bestCls >= 0) {
                    detectionsAboveThreshold++

                    val cx = getValue(0)
                    val cy = getValue(1)
                    val w = getValue(2)
                    val h = getValue(3)

                    val x1 = cx - w / 2f
                    val y1 = cy - h / 2f
                    val x2 = cx + w / 2f
                    val y2 = cy + h / 2f

                    // undo letterbox -> map to original image coords
                    val rx1 = ((x1 - dx) / scale).coerceIn(0f, (srcW - 1).toFloat())
                    val ry1 = ((y1 - dy) / scale).coerceIn(0f, (srcH - 1).toFloat())
                    val rx2 = ((x2 - dx) / scale).coerceIn(0f, (srcW - 1).toFloat())
                    val ry2 = ((y2 - dy) / scale).coerceIn(0f, (srcH - 1).toFloat())

                    boxes.add(Rect2d(rx1.toDouble(), ry1.toDouble(),
                        max(0.0, (rx2 - rx1).toDouble()), max(0.0, (ry2 - ry1).toDouble())))
                    scores.add(bestConf)
                    classes.add(bestCls)

                    Log.d(TAG, "Detection $i: class=${if (bestCls < labels.size) labels[bestCls] else "cls$bestCls"}, conf=$bestConf, box=($rx1,$ry1,$rx2,$ry2)")
                }
            }

            Log.d(TAG, "Found $detectionsAboveThreshold detections above threshold $confThresh")
            Log.d(TAG, "Before NMS: ${boxes.size} detections")

            // apply class-aware NMS
            val keep = Nms.nmsPerClass(boxes, scores, classes, iouThresh.toDouble())

            Log.d(TAG, "After NMS: ${keep.size} detections")

            // draw boxes & labels on rgbaFrame
            for (idx in keep) {
                val r = boxes[idx]
                val x = r.x
                val y = r.y
                val w = r.width
                val h = r.height
                val cls = classes[idx]
                val sc = scores[idx]

                // Draw with thicker, more visible lines
                Imgproc.rectangle(rgbaFrame, Point(x, y), Point(x + w, y + h), Scalar(0.0, 255.0, 0.0, 255.0), 3)
                val label = if (cls in labels.indices) labels[cls] else "cls$cls"

                // Draw background for text
                val textSize = Imgproc.getTextSize(String.format("%s %.2f", label, sc),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, 2, null)
                Imgproc.rectangle(rgbaFrame,
                    Point(x, y - textSize.height - 10),
                    Point(x + textSize.width, y),
                    Scalar(0.0, 255.0, 0.0, 255.0), -1)

                Imgproc.putText(rgbaFrame, String.format("%s %.2f", label, sc),
                    Point(x, y - 5), Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255.0, 255.0, 255.0, 255.0), 2)

                Log.d(TAG, "Drew detection: $label (${String.format("%.2f", sc)}) at ($x,$y,$w,$h)")
            }

            if (keep.isNotEmpty()) {
                Log.i(TAG, "Successfully detected and drew ${keep.size} objects")
            } else {
                Log.w(TAG, "No objects detected in this frame")
            }

            // release mats
            resized.release()
            padded.release()
            roi.release()
            blob.release()
            output.release()

        } catch (e: Exception) {
            Log.e(TAG, "Detection failed: ${e.message}")
            Log.e(TAG, "Exception details: ", e)

            // Draw error message on frame
            Imgproc.putText(rgbaFrame, "Detection Error: ${e.message}",
                Point(10.0, 30.0), Imgproc.FONT_HERSHEY_SIMPLEX,
                0.6, Scalar(0.0, 0.0, 255.0, 255.0), 2)
        }
    }
}