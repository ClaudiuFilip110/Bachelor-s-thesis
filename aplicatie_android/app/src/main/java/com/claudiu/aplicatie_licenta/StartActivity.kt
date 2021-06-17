package com.claudiu.aplicatie_licenta

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import com.claudiu.aplicatie_licenta.ml.ModelAndroid
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer

class StartActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_start)
    }

    fun model() {

        var byteBuffer : ByteBuffer = ByteBuffer.allocate(0)
        val model = ModelAndroid.newInstance(this)

        // Creates inputs for reference.
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 16, 9), DataType.FLOAT32)
        inputFeature0.loadBuffer(byteBuffer)

        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        Log.d("TEXT",outputs.toString())
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        // Releases model resources if no longer used.
        model.close()


    }
}