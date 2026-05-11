from ultralytics import YOLO

print("Loading YOLO model...")
# השורה הזו תוריד אוטומטית את המודל הקל והמהיר ביותר למחשב שלך
model = YOLO('yolo11n.pt') 

print("Analyzing the image...")
# אנחנו נותנים למודל את התמונה שהורדת
results = model(r'D:\isaac\3D_Drone_RL\YOLO\test_image.jpg') 

print("Showing results!")
# הפקודה הזו תפתח את התמונה עם הריבועים שהמודל צייר
results[0].show()