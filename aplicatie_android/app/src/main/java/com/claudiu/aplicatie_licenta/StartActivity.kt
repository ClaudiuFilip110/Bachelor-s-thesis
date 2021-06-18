package com.claudiu.aplicatie_licenta

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import com.claudiu.aplicatie_licenta.ml.ModelAndroid
import kotlinx.android.synthetic.main.activity_start.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.*
import java.util.*

class StartActivity : AppCompatActivity() {

    private val REQUEST_EXTERNAL_STORAGE = 1
    private val PERMISSIONS_STORAGE = arrayOf(
        Manifest.permission.READ_EXTERNAL_STORAGE,
        Manifest.permission.WRITE_EXTERNAL_STORAGE
    )
    private lateinit var text: String

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_start)
        initComponents()
        verifyStoragePermissions(this)
        predict_coin()
    }

    private fun initComponents() {
        app_icon.setImageResource(R.drawable.cryptopred)
    }

    fun verifyStoragePermissions(activity: Activity?) {
        // Check if we have write permission
        val permission = ActivityCompat.checkSelfPermission(
            activity!!,
            Manifest.permission.READ_EXTERNAL_STORAGE
        )
        if (permission != PackageManager.PERMISSION_GRANTED) {
            // We don't have permission so prompt the user
            ActivityCompat.requestPermissions(
                activity,
                PERMISSIONS_STORAGE,
                REQUEST_EXTERNAL_STORAGE
            )
        }
    }

    private fun readFromFile(file: File): Array<Array<FloatArray>> {
        val input: InputStream
        input = FileInputStream(file)
        val a = Array(1) {
            Array(16) {
                FloatArray(9)
            }
        }
        val reader = BufferedReader(InputStreamReader(input))
        for (row in a[0].indices) {
            val linee: String = reader.readLine()
            val lineElems = linee.split(" ")
            for (it in a[0][row].indices) {
                a[0][row][it] = lineElems[it].toFloat()
            }
        }
        input.close()
        return a
    }

    private fun determineOutputFromProbabilites(probabilities: FloatArray): String {
        val maxIdx = probabilities.indices.maxBy { probabilities[it] } ?: -1
        return if (maxIdx == 1)
            "va creste"
        else
            "va scadea"
    }

    private fun modelInference(floatVector: FloatArray): FloatArray {
        val model = ModelAndroid.newInstance(this)

        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 16, 9), DataType.FLOAT32)
        inputFeature0.loadArray(floatVector)

        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        model.close()
        return outputFeature0.floatArray
    }

    private fun shrinkShapeForModel(a: Array<Array<FloatArray>>): FloatArray {
        var floatVector = floatArrayOf()
        for (row in a[0].indices) {
            for (it in a[0][row].indices) {
                floatVector = floatVector.plus(a[0][row][it])
            }
        }
        return floatVector
    }

    private fun predict_coin() {
        if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
            val sdcard = this.filesDir
            val file = File("$sdcard", "INFERENCE_DATA")

            /** deprecated version of code
            var byteBuffer : ByteBuffer =  ByteBuffer.allocate(1*16*9*4)
            for (row in a[0].indices) {
                for (it in a[0][row].indices) {
                    byteBuffer.putFloat(a[0][row][it])
                }
            }
            ....

            inputFeature0.loadBuffer(byteBuffer)
             */

            val a = readFromFile(file)
            var floatVector = shrinkShapeForModel(a)
            val probabilities = modelInference(floatVector)
            val output = determineOutputFromProbabilites(probabilities)

            text = "Pretul criptomonedei $output fata de valoarea initiala"
            tf_output.text = text
        }
    }
}