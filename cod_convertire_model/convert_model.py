import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('C:\Proiecte\Licenta\models\Model-final-arhitectura-mare') # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('cod_convertire_model\model_android.tflite', 'wb') as f:
  f.write(tflite_model)
