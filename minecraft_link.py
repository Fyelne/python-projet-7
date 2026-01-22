from time import time
import time
from mcrcon import MCRcon
from dotenv import load_dotenv
import os

# Charger les variables d'environnement
load_dotenv()

RCON_HOST = os.getenv("RCON_HOST")
RCON_PORT = int(os.getenv("RCON_PORT"))
RCON_PASSWORD = os.getenv("RCON_PASSWORD")

def send_command(command):
    with MCRcon(RCON_HOST, RCON_PASSWORD, port=RCON_PORT) as mcr:
        response = mcr.command(command)
        return response

def test_connection():
    try:
        # Message de bienvenue
        send_command("/say Python est connecté à Minecraft !")
        time.sleep(1)
        
    except Exception as e:
        #print(f"[✗] Erreur: {e}")
        print("Vérifiez que le serveur Minecraft est en cours d'exécution")

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