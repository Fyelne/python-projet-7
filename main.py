"""
Main entry point pour le projet Minecraft + Hand Gesture Recognition
Intègre la reconnaissance de gestes avec la connexion Minecraft
"""

import logging
import threading
import hand as hg
import face as fg  
import minecraft_link as ml

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Démarrage de l'application...")
    
    try:
        # Test de connexion Minecraft
        logger.info("Test de connexion à Minecraft...")
        ml.test_connection()   
    except Exception as e:
        logger.warning(f"Erreur connexion RCON: {e}")
    
    try:
        # Démarrer la reconnaissance de gestes
        logger.info("Initialisation de la reconnaissance de gestes...")
        hg.recognize_hand_gestures() 
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution: {e}")
        

    try:
        # Démarrer la reconnaissance des émotions faciales
        logger.info("Initialisation de la reconnaissance des émotions faciales...")
        fg.recognize_emotions()
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution: {e}")
        raise

    finally:
        logger.info("Fermeture de l'application")


if __name__ == "__main__":
    main()
