import logging
from time import time
import time
from mcrcon import MCRcon
from dotenv import load_dotenv
import os

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

RCON_HOST = os.getenv("RCON_HOST")
RCON_PORT = int(os.getenv("RCON_PORT"))
RCON_PASSWORD = os.getenv("RCON_PASSWORD")

mcr = None

def send_command(command):
        response = mcr.command(command)
        return response

def test_connection():
    global mcr
    retry_count = 0

    while True:
        try:
            mcr = MCRcon(RCON_HOST, RCON_PASSWORD, port=RCON_PORT)
            mcr.connect()
            
            # Envoyer le message de test directement
            mcr.command("/say Python est connecté à Minecraft !")
            time.sleep(1)
            logger.info("[✓] Connexion RCON établie")
            return True
            
        except Exception as e:
            retry_count += 1
            logger.warning(f"[!] Tentative connexion {retry_count} - Nouvelle tentative dans 2 secondes...")
            time.sleep(2)

        
def message(text):
    command = f"/say {text}"
    response = send_command(command)
    return response        

def place_block(x, y, z, block_type="stone"):
    command = f"/setblock {x} {y} {z} {block_type}"
    response = send_command(command)
    return response

def get_player_position():
    command = "/tp @p ~ ~ ~"
    response = send_command(command)
    try:
        parts = response.split()
        x, y, z = map(int, parts[-3:])
        return (x, y, z)
    except:
        return None
    
def move_player(dx, dy, dz):
    command = f"/tp @p ~{dx} ~{dy} ~{dz}"
    response = send_command(command)
    return response

def give_item(item, quantity=1):
    command = f"/give @p {item} {quantity}"
    response = send_command(command)
    return response

def build_igloo(): 
    message("Construction d'une maison igloo...")
    command = f"execute at @p run place structure minecraft:igloo ~5 ~ ~5"
    response = send_command(command)
    return response