@echo off
echo ========================================
echo  Detector de Lengua de Señas Chilena
echo ========================================
echo.

REM Inicialización del entorno virtual
echo Activando entorno virtual...
call .venv\Scripts\activate.bat

REM Verificación de disponibilidad del intérprete de Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: El interprete de Python no se encuentra disponible o no esta configurado en el PATH del sistema.
    pause
    exit /b 1
)

echo Interprete de Python detectado exitosamente. Inicializando servicio...
echo.

REM Transición al directorio raíz de la aplicación web
cd django_app

REM Ejecución de migraciones de la base de datos
echo Sincronizando el esquema de base de datos de Django...
python manage.py migrate --run-syncdb

if errorlevel 1 (
    echo ERROR: Fallo durante la ejecución de migraciones de base de datos.
    echo Por favor, verifique la instalación de las dependencias requeridas (pip install -r ..\requirements.txt)
    pause
    exit /b 1
)

echo.
echo Sincronización de base de datos finalizada con éxito.
echo.
echo El servicio web se encuentra en ejecución en: http://127.0.0.1:8000/
echo Presione Ctrl+C para interrumpir la ejecución del servidor.
echo.

REM Inicialización del servidor de desarrollo de Django
python manage.py runserver 127.0.0.1:8000
pause
