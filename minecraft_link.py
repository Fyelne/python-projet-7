import logging
from time import time
import time
from mcrcon import MCRcon

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
last_move = 0

class Minecraft_link:
    def __init__(self, host, password, port):
        self.host = host
        self.password = password
        self.port = port
        self.rcon = MCRcon(host, password, port)
        self.connected = False
        self._establish_connection()

    def _establish_connection(self):
        try:
            self.rcon.connect()
            self.connected = True
            logger.info("[✓] Connexion RCON établie")
            self.rcon.command("/say Python est connecté à Minecraft !")
            return True
        except Exception as e:
            self.connected = False
            logger.error(f"[✗] Impossible de se connecter au serveur: {e}")
            return False

    def send_command(self, command):
        if not self.connected:
            logger.warning("Pas connecté au serveur Minecraft. Tentative de reconnexion...")
            if not self._establish_connection():
                logger.error("Commande annulée - serveur indisponible.")
                return None
        
        try:
            response = self.rcon.command(command)
            return response
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi de la commande: {e}")
            self.connected = False
            return None

    def test_connection(self):
        if not self.connected:
            if not self._establish_connection():
                return False
        
        try:
            self.rcon.command("/say Python est connecté à Minecraft !")
            logger.info("[✓] Test de connexion réussi")
            return True
        except Exception as e:
            logger.error(f"[✗] Test de connexion échoué: {e}")
            self.connected = False
            return False

    def message(self, text):
        command = f"/say {text}"
        response = self.send_command(command)
        return response        

    def place_block(self, x, y, z, block_type="stone"):
        command = f"/setblock {x} {y} {z} {block_type}"
        response = self.send_command(command)
        return response

    def get_player_position(self):
        command = "/tp @p ~ ~ ~"
        response = self.send_command(command)
        try:
            parts = response.split()
            x, y, z = map(int, parts[-3:])
            return (x, y, z)
        except:
            return None
        
    def move_player(self,dx, dy, dz):
        command = f"/tp @p ~{dx} ~{dy} ~{dz}"
        response = self.send_command(command)
        return response

    def give_item(self, item, quantity=1):
        command = f"/give @p {item} {quantity}"
        response = self.send_command(command)
        return response

    def build_igloo(self): 
        self.message("Construction d'une maison igloo...")
        command = f"execute at @p run place structure minecraft:igloo ~5 ~ ~5"
        response = self.send_command(command)
        return response

    def build_house(self):
        self.message("Construction d'une maison...")
        command = f"execute at @p run place structure minecraft:house ~5 ~ ~5"
        response = self.send_command(command)
        return response


    def move_forward(self, speed=0.25, cooldown=0.05):
        global last_move
        now = time.time()

        if now - last_move > cooldown:
            self.send_command(
                f"execute as @p at @p run tp @p ^ ^ ^{speed}")
            last_move = now
            self._last_move = now
