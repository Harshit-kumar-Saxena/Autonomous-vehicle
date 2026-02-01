# serial communication 
# control the bot manually
#include <Wire.h>
#include <QMC5883LCompass.h>
#include <TinyGPS++.h>
#include <SoftwareSerial.h>

//Compass  
QMC5883LCompass compass;

//GPS Setup 
static const int RX = 4; // GPIO4 = D2 (but used here for GPS RX)
static const int TX = 3; // GPIO3 = RX on ESP8266 (used for GPS TX)
static const uint32_t GPSBaud = 9600;

TinyGPSPlus gps;
SoftwareSerial GPS(RX, TX);  // RX = D4, TX = D3

void setup() {
  Serial.begin(9600);            
  Wire.begin(D2, D1);            // Compass on I2C (D2 = SDA, D1 = SCL)
  GPS.begin(GPSBaud);           

  compass.init();
  compass.setCalibration(-250, 880, -485, 1140, -600, 900);

  Serial.println("Compass and GPS initialized.");
}

void loop() {
  int x, y, z, azimuth;
  char direction[3];

  compass.read();
  x = compass.getX();
  y = compass.getY();
  z = compass.getZ();
  azimuth = compass.getAzimuth();
  compass.getDirection(direction, azimuth);

  Serial.print("Compass => X: "); Serial.print(x);
  Serial.print(" Y: "); Serial.print(y);
  Serial.print(" Z: "); Serial.print(z);
  Serial.print(" Heading: "); Serial.print(azimuth);
  Serial.print("° "); Serial.println(direction);

  while (GPS.available() > 0) {
    gps.encode(GPS.read());
    if (gps.location.isUpdated()) {
      Serial.print("GPS => Latitude: ");
      Serial.print(gps.location.lat(), 6);
      Serial.print(" Longitude: ");
      Serial.println(gps.location.lng(), 6);
    }
  }

  delay(500);
}
