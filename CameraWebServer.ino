#include "WiFi.h"

const char* ssid = "Hack";
const char* password = "12345678";

void setup() {
  Serial.begin(115200);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("Connecting to WiFi...");
    Serial.print("WiFi status code: ");
    Serial.println(WiFi.status());  // Debug output
  }

  Serial.println("Connected to the WiFi network");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  // Add your main code here
}
