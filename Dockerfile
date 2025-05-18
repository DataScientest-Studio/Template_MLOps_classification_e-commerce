# Utiliser l'image python:3.9-slim comme base
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système essentielles
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl git && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de dépendances Python
COPY requirements.txt /app/

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt
#RUN pip install -r requirements.txt

# Copier le code source du projet
COPY . /app/

# Commande par défaut (sera passée au script d'entrée)
# CMD ["python", "src/data/import_raw_data.py"]