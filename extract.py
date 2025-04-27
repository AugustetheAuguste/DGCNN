import os
import csv
from shutil import copyfile

# Chemins
csv_file = './metadata_modelnet40.csv'  # Fichier CSV contenant les métadonnées
base_dir = './ModelNet40'  # Dossier contenant les fichiers .off
output_dir = './ModelNet40_bowl_only'  # Dossier où sauvegarder les fichiers bowl

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(output_dir, exist_ok=True)

# Lire le fichier CSV et copier les fichiers de la classe "bowl"
with open(csv_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['class'] == 'bowl':  # Filtrer uniquement la classe "bowl"
            src_path = os.path.join(base_dir, row['object_path'])
            dest_path = os.path.join(output_dir, os.path.basename(row['object_path']))
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            copyfile(src_path, dest_path)

print(f"Extraction terminée. Les fichiers de la classe 'bowl' sont dans {output_dir}.")