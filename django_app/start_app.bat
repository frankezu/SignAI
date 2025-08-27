@echo off
echo ========================================
echo  Detector de Lengua de Señas Chilena
echo ========================================
echo.

REM Cambiar al directorio de Django
cd /d "c:\Users\xg645\OneDrive\Documentos\Proyecto_ia_legua_se-as_2025\django_app"

REM Verificar que estamos en el directorio correcto
echo Directorio actual: %cd%
echo.

REM Activar entorno virtual y ejecutar migraciones
echo Aplicando migraciones...
"C:\Users\xg645\OneDrive\Documentos\Proyecto_ia_legua_se-as_2025\.venv\Scripts\python.exe" manage.py migrate

if errorlevel 1 (
    echo ERROR: Problema con las migraciones
    pause
    exit /b 1
)

echo.
echo Migraciones aplicadas correctamente!
echo.
echo Iniciando servidor Django...
echo La aplicación estará disponible en: http://127.0.0.1:8000/
echo Presiona Ctrl+C para detener el servidor
echo.

REM Iniciar servidor
"C:\Users\xg645\OneDrive\Documentos\Proyecto_ia_legua_se-as_2025\.venv\Scripts\python.exe" manage.py runserver 127.0.0.1:8000

pause
