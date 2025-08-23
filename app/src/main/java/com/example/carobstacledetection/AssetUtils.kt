package com.example.carobstacledetection

import android.content.Context
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.InputStreamReader

object AssetUtils {
    // copy asset to internal files dir and return path
    fun assetToFile(ctx: Context, assetName: String): String {
        val outFile = File(ctx.filesDir, assetName)
        if (!outFile.exists()) {
            ctx.assets.open(assetName).use { input ->
                FileOutputStream(outFile).use { out ->
                    input.copyTo(out)
                }
            }
        }
        return outFile.absolutePath
    }

    fun readLines(filePath: String): List<String> {
        val file = File(filePath)
        return file.bufferedReader().useLines { it.map { l -> l.trim() }.filter { it.isNotEmpty() }.toList() }
    }
}