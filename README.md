# Gesture Controller - Reconnaissance de Gestes pour Minecraft

Projet de reconnaissance de gestes via caméra pour contrôler des actions dans Minecraft. Dessinez des formes dans l'air avec votre main et le système les reconnaît grâce à l'algorithme $1 Recognizer.

## Fonctionnalités

- 🎥 **Détection de la main en temps réel** via MediaPipe
- ✏️ **Dessin dans l'air** avec geste de pincement (pouce + index)
- 🔍 **Reconnaissance de formes** avec l'algorithme $1 (dollarpy)
- 🎮 **Intégration Minecraft** via protocole RCON

## Installation

### Prérequis

- Python 3.11+
- Webcam

### Linux

```sh
python3.11 -m venv venv  
source venv/bin/activate
pip install -r requirements.txt
```

### Windows

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Utilisation

### 1. Démarrer le serveur Minecraft (optionnel)

#### Linux

```sh
./Minecraft/start.sh
```

#### Windows

```powershell
.\Minecraft\start.bat
```

### 2. Lancer la reconnaissance de gestes

```sh
python main.py
```

### 3. Commandes

| Commande                  | Action                                         |
|---------------------------|------------------------------------------------|
| **Pincement pouce+index** | Démarrer/arrêter le dessin                     |
| `t`                       | Sauvegarder le tracé comme template            |
| `c`                       | Effacer le canvas                              |
| `q`                       | Quitter                                        |

## Optimisations de détection

Le système intègre plusieurs optimisations pour une détection fluide et sans coupures :

### Lissage des positions

- **Moyenne mobile** : Calcul sur les 3 dernières positions pour éliminer le bruit
- **Lissage exponentiel** : Facteur alpha de 0.6 pour des mouvements fluides mais réactifs

### Tolérance aux pertes de détection

Quand MediaPipe perd temporairement la main (occlusion, mouvement rapide) :

- Le système continue le dessin pendant **8 frames** avec la dernière position connue
- Indicateur visuel orange pour montrer la position estimée
- Évite les coupures brutales dans le tracé

### Interpolation pour mouvements rapides

Quand la main bouge très vite :

- Si la distance entre 2 points dépasse **50 pixels**, des points intermédiaires sont automatiquement ajoutés
- Garantit un trait continu même à grande vitesse
- Fréquence d'échantillonnage de **8ms** (~125 points/seconde)

### Hystérésis pour le pincement

- Seuil de démarrage : distance < 0.045
- Seuil d'arrêt : distance > 0.060
- Évite les démarrages/arrêts intempestifs (jitter)

## Paramètres configurables

Dans `hand.py`, vous pouvez ajuster :

| Paramètre                  | Défaut   | Description                       |
|----------------------------|----------|-----------------------------------|
| `W, H`                     | 640, 480 | Résolution de la caméra           |
| `SAMPLE_MS`                | 8        | Intervalle d'échantillonnage (ms) |
| `SMOOTHING_BUFFER_SIZE`    | 3        | Taille du buffer de lissage       |
| `POSITION_SMOOTHING_ALPHA` | 0.6      | Réactivité (0=lisse, 1=brut)      |
| `LOST_FRAMES_TOLERANCE`    | 8        | Frames tolérées sans détection    |
| `MAX_POINT_DISTANCE`       | 50       | Distance max avant interpolation  |
| `PINCH_START_THRESH`       | 0.045    | Seuil de pincement pour démarrer  |
| `PINCH_STOP_THRESH`        | 0.060    | Seuil de pincement pour arrêter   |
| `MIN_POINTS`               | 30       | Points minimum pour reconnaître   |

## Architecture

```md
├── main.py              # Point d'entrée
├── hand.py              # Reconnaissance de gestes
├── minecraft_link.py    # Connexion RCON Minecraft
├── requirements.txt     # Dépendances Python
└── Minecraft/           # Serveur Minecraft Spigot
```

## Dépendances

- `opencv-python` - Capture vidéo et affichage
- `mediapipe` - Détection de la main
- `dollarpy` - Algorithme $1 Recognizer
- `numpy` - Calculs matriciels
- `mcrcon` - Protocole RCON Minecraft
