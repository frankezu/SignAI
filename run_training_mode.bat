@echo off
echo Activando entorno virtual y ejecutando Django...
call .venv\Scripts\activate.bat
cd django_app
echo.
echo Iniciando servidor Django en http://127.0.0.1:8000/
echo Presiona Ctrl+C para detener el servidor
echo.
python manage.py runserver
pause
