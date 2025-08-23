package com.example.carobstacledetection

import android.content.Context
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
    private val inputSize: Int = 640,
    private val confThresh: Float = 0.35f,
    private val iouThresh: Float = 0.45f
) {

    private val net: Net
    private val labels: List<String>
    private val numClasses: Int

    init {
        val onnxPath = AssetUtils.assetToFile(ctx, onnxAssetName)
        val namesPath = AssetUtils.assetToFile(ctx, namesAssetName)
        net = Dnn.readNetFromONNX(onnxPath)
        labels = AssetUtils.readLines(namesPath)
        numClasses = labels.size
    }

    // rgbFrame: RGB Mat, rgbaFrame: RGBA Mat for drawing (display)
    fun detectAndRender(rgbFrame: Mat, rgbaFrame: Mat){
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

        // blob: model expects RGB normalized to [0,1]
        val blob = Dnn.blobFromImage(padded, 1.0 / 255.0, Size(inputSize.toDouble(), inputSize.toDouble()),
            Scalar(0.0, 0.0, 0.0), /*swapRB*/ false, /*crop*/ false)

        net.setInput(blob)
        val output = net.forward() // shape: [1, N, 85] or [1,85,N] depending on export

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

        val boxes = mutableListOf<Rect2d>()
        val scores = mutableListOf<Float>()
        val classes = mutableListOf<Int>()

        for (i in 0 until numPred) {
            // compute base index depending on layout
            // if transposed: value at (c, i) where c in 0..(stride-1)
            // if not transposed: value at (i, c)
            val getValue: (Int) -> Float = { c ->
                if (transposed) {
                    // index: c * numPred + i
                    data[c * numPred + i]
                } else {
                    // index: i * stride + c
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
            if (bestConf < confThresh || bestCls < 0) continue

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
        }

        // apply class-aware NMS
        val keep = Nms.nmsPerClass(boxes, scores, classes, iouThresh.toDouble())

        // draw boxes & labels on rgbaFrame
        for (idx in keep) {
            val r = boxes[idx]
            val x = r.x
            val y = r.y
            val w = r.width
            val h = r.height
            val cls = classes[idx]
            val sc = scores[idx]

            Imgproc.rectangle(rgbaFrame, Point(x, y), Point(x + w, y + h), Scalar(0.0, 255.0, 0.0, 255.0), 2)
            val label = if (cls in labels.indices) labels[cls] else "cls$cls"
            Imgproc.putText(rgbaFrame, String.format("%s %.2f", label, sc),
                Point(x, max(16.0, y - 6.0)), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0.0, 255.0, 0.0, 255.0), 2)

        }

        // release mats
        resized.release()
        padded.release()
        blob.release()
        output.release()
    }
}