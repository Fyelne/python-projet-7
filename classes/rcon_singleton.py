from time import sleep
from mcrcon import MCRcon
from dotenv import load_dotenv
import os

from utils.utils import cooldown

class RCONClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RCONClient, cls).__new__(cls)
            cls._instance.init()
        return cls._instance

    def init(self):
        load_dotenv()

        self.rcon_host = os.getenv("RCON_HOST")
        self.rcon_port = int(os.getenv("RCON_PORT"))
        self.rcon_password = os.getenv("RCON_PASSWORD")
        self.rcon_connected = False

        self.last_weather = "clear"

        self.try_connection()

    def send_command(self, command):
        if not self.rcon_connected:
            print("[!] RCON non connecté - commande ignorée")
            return None

        try:
            with MCRcon(self.rcon_host, self.rcon_password, port=self.rcon_port) as mcr:
                response = mcr.command(command)
                return response
        except Exception as e:
            self.rcon_connected = False
            print(f"[!] Erreur RCON: {e}")
            return None

    def try_connection(self):
        try:
            with MCRcon(self.rcon_host, self.rcon_password, port=self.rcon_port) as mcr:
                mcr.command("/say Python est connecté à Minecraft !")
            self.rcon_connected = True
            print("[✓] Connexion RCON établie")
        except Exception as e:
            self.rcon_connected = False
            print("Vérifiez que le serveur Minecraft est en cours d'exécution")

    def message(self, text):
        command = f"/say {text}"
        return self.send_command(command)

    def place_block(self, x, y, z, block_type="stone"):
        command = f"/setblock {x} {y} {z} {block_type}"
        return self.send_command(command)

    def get_player_position(self):
        command = "/tp @p ~ ~ ~"
        response = self.send_command(command)
        try:
            parts = response.split()
            x, y, z = map(int, parts[-3:])
            return (x, y, z)
        except:
            return None

    def move_player(self, dx, dy, dz):
        command = f"/tp @p ~{dx} ~{dy} ~{dz}"
        return self.send_command(command)

    def give_item(self, item, quantity=1):
        command = f"/give @p {item} {quantity}"
        return self.send_command(command)

    def build_igloo(self): 
        self.message("Construction d'une maison igloo...")
        command = f"execute at @p run place structure minecraft:igloo ~5 ~ ~5"
        return self.send_command(command)
    
    @cooldown(5)
    def weather(self, *args):
        if args is None or self.last_weather == args:
            return

        self.last_weather = args
        self.message("Changement de temps...")
        command = f"/weather {args[0]}"
        response = self.send_command(command)
        return response

