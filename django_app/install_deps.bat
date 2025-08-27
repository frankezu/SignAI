@echo off
echo ========================================
echo  Instalación Manual de Dependencias
echo ========================================
echo.

REM Cambiar al directorio correcto
cd /d "c:\Users\xg645\OneDrive\Documentos\Proyecto_ia_legua_se-as_2025\django_app"

echo Instalando Django...
"C:\Users\xg645\OneDrive\Documentos\Proyecto_ia_legua_se-as_2025\.venv\Scripts\pip.exe" install Django==4.2.7

echo.
echo Instalando OpenCV...
"C:\Users\xg645\OneDrive\Documentos\Proyecto_ia_legua_se-as_2025\.venv\Scripts\pip.exe" install opencv-python==4.8.1.78

echo.
echo Instalando MediaPipe...
"C:\Users\xg645\OneDrive\Documentos\Proyecto_ia_legua_se-as_2025\.venv\Scripts\pip.exe" install mediapipe==0.10.7

echo.
echo Instalando Ultralytics...
"C:\Users\xg645\OneDrive\Documentos\Proyecto_ia_legua_se-as_2025\.venv\Scripts\pip.exe" install ultralytics==8.0.196

echo.
echo Instalando Pillow...
"C:\Users\xg645\OneDrive\Documentos\Proyecto_ia_legua_se-as_2025\.venv\Scripts\pip.exe" install Pillow==10.0.1

echo.
echo Instalando NumPy...
"C:\Users\xg645\OneDrive\Documentos\Proyecto_ia_legua_se-as_2025\.venv\Scripts\pip.exe" install numpy==1.24.3

echo.
echo ¡Instalación completada!
echo Ejecuta check_setup.bat para verificar que todo esté correcto
pause
