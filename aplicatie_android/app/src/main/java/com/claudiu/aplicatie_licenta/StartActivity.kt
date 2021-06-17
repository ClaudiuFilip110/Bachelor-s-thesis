package com.claudiu.aplicatie_licenta

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.claudiu.aplicatie_licenta.ml.ModelAndroid
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.io.FileInputStream
import java.io.InputStream
import java.nio.ByteBuffer


class StartActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_start)
        model()
    }

    fun model() {
        var path = applicationContext.filesDir.absolutePath
        var filePath = "$path/ml/text.txt"
        FileInputStream(filePath).use { input ->
            var content: Int
            while (input.read().also { content = it } != -1) {
                Log.d("TEXT", content.toString())
            }
        }

//            var input: InputStream = ""


//            var byteBuffer: ByteBuffer = ByteBuffer.allocate(input.available())
//            val model = ModelAndroid.newInstance(this)
//
//            // Creates inputs for reference.
//            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 16, 9), DataType.FLOAT32)
//            inputFeature0.loadBuffer(byteBuffer)
//
//            // Runs model inference and gets result.
//            val outputs = model.process(inputFeature0)
//            Log.d("TEXT", outputs.toString())
//            val outputFeature0 = outputs.outputFeature0AsTensorBuffer
//
//            // Releases model resources if no longer used.
//            model.close()


    }
}