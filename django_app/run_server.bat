@echo off
echo ========================================
echo  Detector de Lengua de Señas Chilena
echo ========================================
echo.

echo Verificando dependencias...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no está instalado o no está en el PATH
    pause
    exit /b 1
)

echo Python encontrado. Iniciando aplicación...
echo.

REM Aplicar migraciones si es necesario
echo Aplicando migraciones de Django...
python manage.py migrate --run-syncdb

if errorlevel 1 (
    echo ERROR: No se pudieron aplicar las migraciones
    echo Asegúrate de estar en el directorio correcto y tener Django instalado
    pause
    exit /b 1
)

echo.
echo Iniciando servidor de desarrollo...
echo.
echo La aplicación estará disponible en: http://127.0.0.1:8000/
echo Presiona Ctrl+C para detener el servidor
echo.

REM Iniciar servidor de Django
python manage.py runserver 127.0.0.1:8000
