package com.example.carobstacledetection

import org.opencv.core.Rect2d
import kotlin.math.max
import kotlin.math.min

object Nms {
    private fun iou(a: Rect2d, b: Rect2d): Double {
        val x1 = max(a.x, b.x)
        val y1 = max(a.y, b.y)
        val x2 = min(a.x + a.width, b.x + b.width)
        val y2 = min(a.y + a.height, b.y + b.height)
        val inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        val areaA = a.width * a.height
        val areaB = b.width * b.height
        val denom = areaA + areaB - inter
        return if (denom <= 0.0) 0.0 else inter / denom
    }

    // Per-class NMS; returns selected indices (into boxes/scores/classes lists)
    fun nmsPerClass(boxes: List<Rect2d>, scores: List<Float>, classes: List<Int>, iouThresh: Double): List<Int> {
        val byCls = mutableMapOf<Int, MutableList<Int>>()
        for (i in boxes.indices) {
            val cls = classes[i]
            byCls.computeIfAbsent(cls) { mutableListOf() }.add(i)
        }
        val keep = mutableListOf<Int>()
        for ((_, idxs) in byCls) {
            idxs.sortByDescending { scores[it] }
            val removed = BooleanArray(idxs.size)
            for (i in idxs.indices) {
                if (removed[i]) continue
                val a = idxs[i]
                keep.add(a)
                for (j in i + 1 until idxs.size) {
                    if (removed[j]) continue
                    val b = idxs[j]
                    if (iou(boxes[a], boxes[b]) > iouThresh) removed[j] = true
                }
            }
        }
        return keep
    }
}