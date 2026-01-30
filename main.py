"""
Main entry point pour le projet Minecraft + Hand Gesture Recognition
Intègre la reconnaissance de gestes avec la connexion Minecraft
"""

import logging
import os
import threading
from dotenv import load_dotenv
import body
import hand as hg
import minecraft_link as ml

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()
RCON_HOST = os.getenv("RCON_HOST")
RCON_PORT = int(os.getenv("RCON_PORT"))
RCON_PASSWORD = os.getenv("RCON_PASSWORD")


def main():
    logger.info("Démarrage de l'application...")
    
    # Créer et connecter au serveur Minecraft
    logger.info("Connexion à Minecraft...")
    try:
        mc_connect = ml.Minecraft_link(RCON_HOST, RCON_PASSWORD, RCON_PORT)
        if mc_connect.connected:
            connection_thread = threading.Thread(target=mc_connect.test_connection, daemon=True)
            connection_thread.start()
        else:
            logger.warning("Serveur Minecraft indisponible. Continuant sans Minecraft...")
    except Exception as e:
        logger.error(f"Impossible de créer la connexion: {e}")
        mc_connect = None
    
    try:
        # Démarrer la reconnaissance de gestes
        logger.info("Initialisation de la reconnaissance de gestes...")
        hg.recognize_hand_gestures(mc_connect)
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution: {e}")
        raise

    try:
        # Démarrer la reconnaissance du corps
        logger.info("Initialisation de la reconnaissance du corps...")
        body.recognize_body_gestures(mc_connect)
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution: {e}")
        raise

    finally:
        logger.info("Fermeture de l'application")


if __name__ == "__main__":
    main()
