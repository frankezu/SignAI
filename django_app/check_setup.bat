@echo off
echo Verificando instalación de Django...

REM Cambiar al directorio correcto
cd /d "c:\Users\xg645\OneDrive\Documentos\Proyecto_ia_legua_se-as_2025\django_app"

REM Verificar Django
"C:\Users\xg645\OneDrive\Documentos\Proyecto_ia_legua_se-as_2025\.venv\Scripts\python.exe" -c "import django; print(f'Django version: {django.get_version()}')"

if errorlevel 1 (
    echo ERROR: Django no está instalado correctamente
    pause
    exit /b 1
)

echo.
echo Verificando otras dependencias...
"C:\Users\xg645\OneDrive\Documentos\Proyecto_ia_legua_se-as_2025\.venv\Scripts\python.exe" -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
"C:\Users\xg645\OneDrive\Documentos\Proyecto_ia_legua_se-as_2025\.venv\Scripts\python.exe" -c "import mediapipe; print('MediaPipe: OK')"
"C:\Users\xg645\OneDrive\Documentos\Proyecto_ia_legua_se-as_2025\.venv\Scripts\python.exe" -c "import ultralytics; print('Ultralytics: OK')"

echo.
echo Todas las dependencias están instaladas correctamente!
echo.

REM Ejecutar migraciones
echo Aplicando migraciones de Django...
"C:\Users\xg645\OneDrive\Documentos\Proyecto_ia_legua_se-as_2025\.venv\Scripts\python.exe" manage.py migrate

echo.
echo ¡Listo! Ahora puedes ejecutar start_app.bat para iniciar la aplicación
pause
