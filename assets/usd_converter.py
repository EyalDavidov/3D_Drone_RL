from pxr import Usd, UsdGeom

# טעינת קובץ ה-USD שלך
stage = Usd.Stage.Open(r"D:\isaac\3D_Drone_RL\assets\room_with_poles.usd")

# מעבר על כל האובייקטים (Prims) בסצנה
print("--- מחפש עמודים במפה ---")
for prim in stage.Traverse():
    # בדיקה אם שם האובייקט מכיל את המילה pole
    if "pole" in prim.GetName().lower():
        # שליפת נתוני המיקום התלת-ממדי
        xform = UsdGeom.Xformable(prim)
        transform_matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = transform_matrix.ExtractTranslation()
        
        print(f"pole name: {prim.GetName()}")
        print(f"coordinates (X, Y, Z): ({translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f})\n")