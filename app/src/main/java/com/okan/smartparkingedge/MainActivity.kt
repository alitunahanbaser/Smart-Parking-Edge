package com.okan.smartparkingedge

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Collections
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var tvAvailable: TextView
    private lateinit var tvEntryExit: TextView
    private lateinit var etCapacity: EditText
    private lateinit var btnSetCap: Button
    private lateinit var btnMinus: Button
    private lateinit var btnPlus: Button
    private lateinit var previewView: PreviewView
    private lateinit var cameraContainer: FrameLayout
    private lateinit var virtualLine: View
    private lateinit var seekLine: SeekBar
    private lateinit var btnToggle: Button
    private lateinit var tvLogs: TextView

    private var totalCapacity = 100
    private var occupiedCount = 0
    private var entryCount = 0
    private var exitCount = 0
    private var lineRatio = 0.55f
    private var isMonitoring = false
    private var isProcessingFrame = false

    private var pythonEngine: PyObject? = null
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private val textRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val logHistory = mutableListOf<String>()

    // 10 saniyelik plaka bazlı cooldown ve içerideki araçların giriş zamanı hafızası
    private val plateCooldownMap = mutableMapOf<String, Long>()
    private val activeVehicles = mutableMapOf<String, Long>()
    private val COOLDOWN_MS = 10_000L

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) startCamera()
            else Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvAvailable = findViewById(R.id.tvAvailable)
        tvEntryExit = findViewById(R.id.tvEntryExit)
        etCapacity = findViewById(R.id.etCapacity)
        btnSetCap = findViewById(R.id.btnSetCap)
        btnMinus = findViewById(R.id.btnMinus)
        btnPlus = findViewById(R.id.btnPlus)
        previewView = findViewById(R.id.previewView)
        cameraContainer = findViewById(R.id.cameraContainer)
        virtualLine = findViewById(R.id.virtualLine)
        seekLine = findViewById(R.id.seekLine)
        btnToggle = findViewById(R.id.btnToggle)
        tvLogs = findViewById(R.id.tvLogs)

        loadState()
        updateUI()

        Thread {
            try {
                if (!Python.isStarted()) {
                    Python.start(AndroidPlatform(this))
                }
                val py = Python.getInstance()
                pythonEngine = py.getModule("parking_engine")

                val env = OrtEnvironment.getEnvironment()
                val modelBytes = assets.open("yolov8n.onnx").readBytes()
                val session = env.createSession(modelBytes, OrtSession.SessionOptions())
                ortEnv = env
                ortSession = session

                runOnUiThread {
                    Toast.makeText(this, "AI Engine Ready", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                e.printStackTrace()
                runOnUiThread {
                    Toast.makeText(this, "Init Error: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }.start()

        btnSetCap.setOnClickListener {
            val newCap = etCapacity.text.toString().toIntOrNull()
            if (newCap != null && newCap > 0) {
                totalCapacity = newCap
                saveState()
                updateUI()
            }
        }

        btnPlus.setOnClickListener {
            if (occupiedCount < totalCapacity) {
                occupiedCount++
                saveState()
                updateUI()
            }
        }

        btnMinus.setOnClickListener {
            if (occupiedCount > 0) {
                occupiedCount--
                saveState()
                updateUI()
            }
        }

        seekLine.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                lineRatio = (progress + 10) / 100f
                updateLineVisual()
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        cameraContainer.post { updateLineVisual() }

        btnToggle.setOnClickListener {
            isMonitoring = !isMonitoring
            if (isMonitoring) {
                btnToggle.text = "STOP"
                btnToggle.setBackgroundColor(0xFFF44336.toInt())
            } else {
                btnToggle.text = "START"
                btnToggle.setBackgroundColor(0xFF4CAF50.toInt())
            }
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun updateLineVisual() {
        val containerHeight = cameraContainer.height
        if (containerHeight > 0) {
            virtualLine.translationY = containerHeight * lineRatio
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                        processImageProxy(imageProxy)
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processImageProxy(imageProxy: ImageProxy) {
        val pyEngine = pythonEngine
        val env = ortEnv
        val session = ortSession

        if (!isMonitoring || pyEngine == null || env == null || session == null || isProcessingFrame) {
            imageProxy.close()
            return
        }

        isProcessingFrame = true
        try {
            val bitmap = imageProxy.toBitmap()
            val rotated = rotateBitmap(bitmap, imageProxy.imageInfo.rotationDegrees.toFloat())
            val stream = ByteArrayOutputStream()
            rotated.compress(Bitmap.CompressFormat.JPEG, 70, stream)
            val jpegBytes = stream.toByteArray()

            // 1. Python ile görüntüyü 320x320 tensöre çevir
            val prepList = pyEngine.callAttr("preprocess_frame", jpegBytes).asList()
            val tensorBytes = prepList[0].toJava(ByteArray::class.java)
            val origW = prepList[1].toInt()
            val origH = prepList[2].toInt()

            val floatBuffer = ByteBuffer.wrap(tensorBytes)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
            val inputTensor = OnnxTensor.createTensor(
                env, floatBuffer, longArrayOf(1, 3, 320, 320)
            )

            // 2. ONNX Runtime ile YOLOv8n çalıştır
            val inputName = session.inputNames.iterator().next()
            val outputs = session.run(Collections.singletonMap(inputName, inputTensor))
            val rawOutTensor = outputs[0] as OnnxTensor
            val outFloatBuf = rawOutTensor.floatBuffer
            val outBytes = ByteArray(outFloatBuf.remaining() * 4)
            ByteBuffer.wrap(outBytes).order(ByteOrder.nativeOrder()).asFloatBuffer().put(outFloatBuf)

            inputTensor.close()
            outputs.close()

            // 3. Python ile NMS ve sanal çizgi geçiş sayımı yap
            val resStr = pyEngine.callAttr(
                "postprocess_detections", outBytes, origW, origH, lineRatio
            ).toString()

            val json = JSONObject(resStr)
            val newEntry = json.optInt("entry", entryCount)
            val newExit = json.optInt("exit", exitCount)
            val eventObj = json.optJSONObject("event")

            val entryDiff = (newEntry - entryCount).coerceAtLeast(0)
            val exitDiff = (newExit - exitCount).coerceAtLeast(0)

            if (eventObj != null) {
                val dir = eventObj.getString("direction")
                val time = eventObj.getString("time")
                val cropArr = eventObj.getJSONArray("crop")

                val x1 = cropArr.getInt(0).coerceIn(0, rotated.width - 1)
                val y1 = cropArr.getInt(1).coerceIn(0, rotated.height - 1)
                val x2 = cropArr.getInt(2).coerceIn(x1 + 1, rotated.width)
                val y2 = cropArr.getInt(3).coerceIn(y1 + 1, rotated.height)

                val croppedPlate = Bitmap.createBitmap(rotated, x1, y1, x2 - x1, y2 - y1)
                val scaledPlate = Bitmap.createScaledBitmap(
                    croppedPlate,
                    croppedPlate.width * 2,
                    croppedPlate.height * 2,
                    true
                )
                val inputImage = InputImage.fromBitmap(scaledPlate, 0)
                val trPlateRegex = Regex("(0[1-9]|[1-7][0-9]|8[01])[A-Z]{1,3}[0-9]{2,4}")

                textRecognizer.process(inputImage)
                    .addOnSuccessListener { visionText ->
                        var cleaned = visionText.text
                            .uppercase()
                            .filter { it.isLetterOrDigit() }

                        if (cleaned.startsWith("TR")) {
                            cleaned = cleaned.removePrefix("TR")
                        } else if (cleaned.startsWith("T") && cleaned.length > 1 && cleaned[1].isDigit()) {
                            cleaned = cleaned.removePrefix("T")
                        }

                        val firstDigitIdx = cleaned.indexOfFirst { it.isDigit() }
                        if (firstDigitIdx > 0) {
                            cleaned = cleaned.substring(firstDigitIdx)
                        }

                        val matchedPlate = if (cleaned.length in 6..9) {
                            trPlateRegex.find(cleaned)?.value
                        } else {
                            null
                        }

                        val finalPlate = matchedPlate ?: "UNKNOWN"
                        val nowMs = System.currentTimeMillis()

                        // 10 saniye plaka cooldown kontrolü
                        if (finalPlate != "UNKNOWN") {
                            val lastTime = plateCooldownMap[finalPlate] ?: 0L
                            if (nowMs - lastTime < COOLDOWN_MS) {
                                return@addOnSuccessListener
                            }
                            plateCooldownMap[finalPlate] = nowMs
                        }

                        // Sayaçları güncelle
                        if (entryDiff > 0) {
                            entryCount += entryDiff
                            occupiedCount = (occupiedCount + entryDiff).coerceAtMost(totalCapacity)
                        }
                        if (exitDiff > 0) {
                            exitCount += exitDiff
                            occupiedCount = (occupiedCount - exitDiff).coerceAtLeast(0)
                        }

                        // İçerideki araç hafızası ve kalma süresi hesabı
                        var logSuffix = ""
                        if (finalPlate != "UNKNOWN") {
                            if (dir == "ENTRY") {
                                activeVehicles[finalPlate] = nowMs
                            } else if (dir == "EXIT") {
                                val enteredAt = activeVehicles.remove(finalPlate)
                                if (enteredAt != null) {
                                    val stayMinutes = ((nowMs - enteredAt) / 60_000L).coerceAtLeast(1)
                                    logSuffix = " (${stayMinutes} dk)"
                                }
                            }
                        }

                        logHistory.add(0, "[$time] $dir -> $finalPlate$logSuffix")
                        if (logHistory.size > 30) logHistory.removeAt(logHistory.lastIndex)
                        tvLogs.text = logHistory.joinToString("\n")
                        saveState()
                        updateUI()
                    }
                    .addOnFailureListener {
                        if (entryDiff > 0) {
                            entryCount += entryDiff
                            occupiedCount = (occupiedCount + entryDiff).coerceAtMost(totalCapacity)
                        }
                        if (exitDiff > 0) {
                            exitCount += exitDiff
                            occupiedCount = (occupiedCount - exitDiff).coerceAtLeast(0)
                        }
                        logHistory.add(0, "[$time] $dir -> UNKNOWN")
                        if (logHistory.size > 30) logHistory.removeAt(logHistory.lastIndex)
                        tvLogs.text = logHistory.joinToString("\n")
                        saveState()
                        updateUI()
                    }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            isProcessingFrame = false
            imageProxy.close()
        }
    }

    private fun rotateBitmap(source: Bitmap, angle: Float): Bitmap {
        if (angle == 0f) return source
        val matrix = Matrix()
        matrix.postRotate(angle)
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
    }

    private fun saveState() {
        val prefs = getSharedPreferences("SmartParkingPrefs", Context.MODE_PRIVATE)
        prefs.edit()
            .putInt("totalCapacity", totalCapacity)
            .putInt("occupiedCount", occupiedCount)
            .apply()
    }

    private fun loadState() {
        val prefs = getSharedPreferences("SmartParkingPrefs", Context.MODE_PRIVATE)
        totalCapacity = prefs.getInt("totalCapacity", 100)
        occupiedCount = prefs.getInt("occupiedCount", 0)
        etCapacity.setText(totalCapacity.toString())
    }

    private fun updateUI() {
        val available = (totalCapacity - occupiedCount).coerceAtLeast(0)
        tvAvailable.text = "$available / $totalCapacity"
        tvEntryExit.text = "IN: $entryCount  |  OUT: $exitCount"
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        ortSession?.close()
        ortEnv?.close()
    }
}