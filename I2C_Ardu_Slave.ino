#include <Wire.h>

#define SLAVE_ADDRESS 0x42 //0x42 for Pixhawk 0x49 for Raspberry
int number = 0;
int state = 0;
int a=0;

void setup() {
pinMode(13, OUTPUT);
Serial.begin(9600); // start serial for output
// initialize i2c as slave
Wire.begin(SLAVE_ADDRESS);

// define callbacks for i2c communication
 Wire.onReceive(receiveData);
Wire.onRequest(sendData);

Serial.println("Ready!");
}

void loop() {
delay(100);

}

// callback for received data
void receiveData(int byteCount){

while(Wire.available()) {
number = Wire.read();
Serial.print("data received: ");
Serial.println(number);

if (number == 1){

if (state == 0){
digitalWrite(13, HIGH); // set the LED on
state = 1;
}
else{
digitalWrite(13, LOW); // set the LED off
state = 0;
}
}
}
}

// callback for sending data
void sendData(){
Wire.write(1);
Wire.write(1);
Wire.write(1);
Wire.write(1);
Wire.write(1);
Wire.write(1);
Wire.write(255);//optmx
Wire.write(0);
Wire.write(0);
Wire.write(0);
Wire.write(1);
Wire.write(1);
Wire.write(1);
Wire.write(1);
Wire.write(1);
Wire.write(1);
Wire.write(1);
Wire.write(1);
Wire.write(1);
Wire.write(1);
Wire.write(127);
Wire.write(100);
}
