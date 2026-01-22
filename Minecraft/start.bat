@echo off
title Serveur Minecraft

:: Allocation mémoire (modifiable)
set RAM_MIN=4G
set RAM_MAX=6G

:: Nom du fichier jar
set JAR=server.jar

echo ===============================
echo   Lancement du serveur Minecraft
echo ===============================
echo.

java -Xms%RAM_MIN% -Xmx%RAM_MAX% -jar %JAR% nogui

echo.
echo Serveur arrêté.
pause
