/*
 * =====================================================================
 *  LD7182 - AI for IoT
 *  AI-Based Smart Agriculture Monitoring & Crop Recommendation
 *  Hardware  : ESP32-S3 DevKitC-1
 *  AI        : TensorFlow Lite Micro (INT8 quantised, ~5.5 KB)
 *  Cloud     : Blynk IoT
 *  Sensors   : DHT22, Soil Moisture (analogue), Rain Sensor (analogue)
 *  Actuators : 5V Relay (water pump), Buzzer, I2C LCD 16x2
 * =====================================================================
 */

#define BLYNK_TEMPLATE_ID   "TMPL3xxxxxxxx"
#define BLYNK_TEMPLATE_NAME "SmartFarm"
#define BLYNK_AUTH_TOKEN    "YOUR_BLYNK_AUTH_TOKEN_HERE"
#define BLYNK_PRINT Serial

#include <WiFi.h>
#include <BlynkSimpleEsp32.h>
#include <DHT.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// ---- TensorFlow Lite Micro ----
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "crop_model_data.h"           // INT8 model auto-generated

// =========================== PIN MAP ================================
#define DHT_PIN         4         // GPIO4  -> DHT22 data
#define DHT_TYPE        DHT22
#define SOIL_PIN        34        // GPIO34 -> Soil moisture analogue
#define RAIN_PIN        35        // GPIO35 -> Rain sensor analogue
#define RELAY_PIN       26        // GPIO26 -> Relay (active LOW)
#define BUZZER_PIN      27        // GPIO27 -> Buzzer
#define LED_STATUS_PIN  2         // Built-in status LED
#define I2C_SDA         21
#define I2C_SCL         22

// =========================== CONFIG =================================
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";

const int   SOIL_THRESHOLD_PCT   = 30;     // pump ON when soil moisture < 30%
const unsigned long SAMPLE_INTERVAL_MS = 5000UL;

// Scaler parameters exported from Python (StandardScaler)
const float FEATURE_MEAN[7]  = {50.5518f, 53.3627f, 48.1491f, 25.6164f,
                                71.4818f,  6.4695f, 103.4637f};
const float FEATURE_SCALE[7] = {36.9173f, 32.9859f, 50.6470f,  5.0639f,
                                22.2638f,  0.7739f,  54.9584f};

// Class labels (must match LabelEncoder ordering in training script)
const char* CROP_NAMES[22] = {
  "apple","banana","blackgram","chickpea","coconut","coffee",
  "cotton","grapes","jute","kidneybeans","lentil","maize",
  "mango","mothbeans","mungbean","muskmelon","orange","papaya",
  "pigeonpeas","pomegranate","rice","watermelon"
};

// =========================== GLOBALS ================================
DHT dht(DHT_PIN, DHT_TYPE);
LiquidCrystal_I2C lcd(0x27, 16, 2);
BlynkTimer timer;

bool   pumpManualOverride = false;
bool   pumpState          = false;
String currentCrop        = "—";

// ---- TFLM globals ----
namespace {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input  = nullptr;
  TfLiteTensor* output = nullptr;
  constexpr int kTensorArenaSize = 16 * 1024;     // 16 KB arena (plenty)
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}

// =========================== TFLM SETUP =============================
void initTinyML() {
  model = tflite::GetModel(crop_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("[TFLM] Model schema mismatch!");
    while (1) { delay(1000); }
  }
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus s = interpreter->AllocateTensors();
  if (s != kTfLiteOk) {
    Serial.println("[TFLM] AllocateTensors() failed!");
    while (1) { delay(1000); }
  }
  input  = interpreter->input(0);
  output = interpreter->output(0);

  Serial.printf("[TFLM] Input dims: %d, type: %d (INT8)\n",
                input->dims->data[1], input->type);
  Serial.printf("[TFLM] Output dims: %d, type: %d (INT8)\n",
                output->dims->data[1], output->type);
  Serial.printf("[TFLM] Arena used (approx): %d bytes\n",
                interpreter->arena_used_bytes());
}

// =========================== INFERENCE ==============================
int runInference(float N, float P, float K,
                 float t, float h, float ph, float rain,
                 float& latency_ms) {
  // 1. Scale features (StandardScaler equivalent)
  float feats[7] = {N, P, K, t, h, ph, rain};
  float scaled[7];
  for (int i = 0; i < 7; i++) {
    scaled[i] = (feats[i] - FEATURE_MEAN[i]) / FEATURE_SCALE[i];
  }

  // 2. Quantise to INT8 using input tensor quantisation params
  const float in_scale  = input->params.scale;
  const int   in_zp     = input->params.zero_point;
  for (int i = 0; i < 7; i++) {
    int32_t q = (int32_t)round(scaled[i] / in_scale) + in_zp;
    if (q > 127)  q = 127;
    if (q < -128) q = -128;
    input->data.int8[i] = (int8_t)q;
  }

  // 3. Invoke
  unsigned long t0 = micros();
  TfLiteStatus s = interpreter->Invoke();
  latency_ms = (micros() - t0) / 1000.0f;
  if (s != kTfLiteOk) {
    Serial.println("[TFLM] Invoke failed!");
    return -1;
  }

  // 4. Argmax over output (INT8)
  int   best_idx = 0;
  int8_t best_val = output->data.int8[0];
  for (int i = 1; i < 22; i++) {
    if (output->data.int8[i] > best_val) {
      best_val = output->data.int8[i];
      best_idx = i;
    }
  }
  return best_idx;
}

// =========================== SENSORS ================================
float readSoilPct() {
  int raw = analogRead(SOIL_PIN);            // 0-4095 (ESP32 12-bit ADC)
  // Capacitive: 0 (wet) -> 4095 (dry). Adjust to your hardware.
  float pct = map(raw, 4095, 1500, 0, 100);
  if (pct < 0)   pct = 0;
  if (pct > 100) pct = 100;
  return pct;
}

float readRainfallMmEquivalent() {
  int raw = analogRead(RAIN_PIN);
  // Higher value = drier. Map to a 0-300 mm scale for ML input.
  float mm = map(raw, 4095, 1000, 0, 300);
  if (mm < 0) mm = 0;
  return mm;
}

// =========================== BLYNK HANDLERS =========================
BLYNK_WRITE(V10) {        // manual pump override switch
  pumpManualOverride = (param.asInt() == 1);
  if (pumpManualOverride) {
    digitalWrite(RELAY_PIN, LOW);    // pump ON
    pumpState = true;
  } else {
    digitalWrite(RELAY_PIN, HIGH);   // pump OFF
    pumpState = false;
  }
  Blynk.virtualWrite(V5, pumpState ? "MANUAL ON" : "MANUAL OFF");
}

// =========================== MAIN LOOP TASK =========================
void sampleAndInfer() {
  float t  = dht.readTemperature();
  float h  = dht.readHumidity();
  if (isnan(t) || isnan(h)) { t = 25.0; h = 70.0; }

  float sm = readSoilPct();
  float ph = 6.5;                              // demo / placeholder pH
  float rn = readRainfallMmEquivalent();

  // N/P/K demo defaults (could be read from NPK sensor if attached)
  float N = 60.0, P = 50.0, K = 40.0;

  float lat_ms = 0.0f;
  int idx = runInference(N, P, K, t, h, ph, rn, lat_ms);
  if (idx >= 0) currentCrop = CROP_NAMES[idx];

  // Auto irrigation
  if (!pumpManualOverride) {
    if (sm < SOIL_THRESHOLD_PCT) {
      digitalWrite(RELAY_PIN, LOW);  pumpState = true;
      tone(BUZZER_PIN, 1500, 120);
    } else {
      digitalWrite(RELAY_PIN, HIGH); pumpState = false;
    }
  }

  // Push to Blynk
  Blynk.virtualWrite(V0, t);
  Blynk.virtualWrite(V1, h);
  Blynk.virtualWrite(V2, sm);
  Blynk.virtualWrite(V3, ph);
  Blynk.virtualWrite(V4, currentCrop);
  Blynk.virtualWrite(V5, pumpState ? "ON" : "OFF");
  Blynk.virtualWrite(V6, lat_ms);

  // Local LCD
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.printf("T:%4.1fC H:%3.0f%%", t, h);
  lcd.setCursor(0, 1);
  lcd.printf("%-10s P:%s", currentCrop.c_str(), pumpState ? "ON" : "OFF");

  // Serial monitor log
  Serial.printf("[%lu ms] T=%.1f H=%.1f SM=%.0f%% Rain=%.0f -> CROP=%-12s "
                "Pump=%s  inf=%.2f ms\n",
                millis(), t, h, sm, rn, currentCrop.c_str(),
                pumpState ? "ON" : "OFF", lat_ms);

  if (sm < 20) Blynk.logEvent("low_moisture", "Critical: soil < 20%");
}

// =========================== SETUP ==================================
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n==== Smart Agriculture System Booting ====");

  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, HIGH);           // pump OFF (active LOW)
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_STATUS_PIN, OUTPUT);

  Wire.begin(I2C_SDA, I2C_SCL);
  lcd.init(); lcd.backlight();
  lcd.setCursor(0, 0); lcd.print("Smart Farm");
  lcd.setCursor(0, 1); lcd.print("Booting...");

  dht.begin();
  Serial.println("[INIT] DHT22 ready");

  initTinyML();
  Serial.println("[INIT] TinyML model loaded (INT8, 5.5 KB)");

  Blynk.begin(BLYNK_AUTH_TOKEN, WIFI_SSID, WIFI_PASS);
  Serial.println("[INIT] Blynk connected");

  timer.setInterval(SAMPLE_INTERVAL_MS, sampleAndInfer);
  digitalWrite(LED_STATUS_PIN, HIGH);
}

void loop() {
  Blynk.run();
  timer.run();
}
