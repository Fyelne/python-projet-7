#!/bin/bash

# Nom de la fenêtre (optionnel, dépend du terminal)
echo -ne "\033]0;Serveur Minecraft\007"

# Allocation mémoire (modifiable)
RAM_MIN="4G"
RAM_MAX="6G"

# Nom du fichier jar
JAR="server.jar"

echo "==============================="
echo "  Lancement du serveur Minecraft"
echo "==============================="
echo

java -Xms${RAM_MIN} -Xmx${RAM_MAX} -jar "${JAR}" nogui

echo
echo "Serveur arrêté."
read -p "Appuyez sur Entrée pour quitter..."
