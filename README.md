# Feuille-diagnostic
# Leaf Diagnostic — Classification des feuilles de pomme de terre par CNN

Application web qui détecte automatiquement si une feuille de pomme de terre est saine ou atteinte d'une maladie (mildiou précoce ou tardif), à partir d'une simple photo. Le modèle de Deep Learning tourne directement dans le navigateur grâce à TensorFlow.js, sans serveur ni envoi de données.

**Démo en ligne :** https://tonpseudo.github.io/leaf-diagnostic/

---

## Aperçu

L'application accepte une image de feuille au format PNG, JPG, JPEG, BMP ou WEBP, la redimensionne en 128x128 pixels, la passe dans un réseau de neurones convolutif et renvoie en quelques millisecondes :

- La classe prédite (Saine ou Malade)
- La confiance du modèle
- Les probabilités détaillées pour chaque classe

Un mode batch permet d'analyser plusieurs images d'un coup.

---

## Contexte du projet

Mini-projet réalisé dans le cadre de l'évaluation sommative UA3 du cours d'Intelligence Artificielle au **Collège La Cité**.

L'agriculture de précision s'appuie de plus en plus sur des modèles de vision par ordinateur pour détecter les maladies des cultures à grande échelle. Ce projet illustre le cycle complet : préparation des données, conception d'un CNN, entraînement, évaluation, optimisation, puis déploiement sous forme d'application web utilisable par n'importe qui.

### Membres de l'équipe

| # | Nom complet |
|---|-------------|
| 1 | Félicité Djieukam Mouatcho |
| 2 | Josué Joachim |
| 3 | Leila Toundji Kouougue |
| 4 | Gilyns E. Dorsaint |
| 5 | Ourega Isidore Dago |

---

## Le modèle en bref

### Architecture

Réseau de neurones convolutif construit *from scratch*, organisé en quatre blocs convolutifs avec du Dropout réparti à plusieurs niveaux pour limiter le surapprentissage.

```
Conv2D(32)  -> MaxPool -> Dropout(0.25)
Conv2D(64)  -> MaxPool -> Dropout(0.25)
Conv2D(128) -> MaxPool -> Dropout(0.30)
Conv2D(256) -> MaxPool -> Dropout(0.30)
Flatten -> Dense(256) -> Dropout(0.50) -> Dense(1, sigmoid)
```

### Configuration retenue

- **Optimiseur** : Adam (learning rate de 0,001)
- **Fonction de perte** : binary crossentropy
- **Taille des images** : 128x128 pixels, normalisées entre 0 et 1
- **Batch size** : 32
- **EarlyStopping** : patience de 7 epochs sur la val_loss, plafond de 30 epochs

### Performances

Évaluation sur 180 images de validation (20% du dataset) :

| Métrique | Valeur |
|----------|--------|
| Accuracy | 98,3 % |
| F1-score | 0,98 |
| Précision (Diseased) | 1,00 |
| Rappel (Diseased) | 0,97 |
| Précision (Healthy) | 0,95 |
| Rappel (Healthy) | 1,00 |

### Dataset

[Potato Leaf Disease Dataset](https://www.kaggle.com/datasets/muhammadardiputra/potato-leaf-disease-dataset) — 1500 images réparties à l'origine en trois classes (*Healthy*, *Early Blight*, *Late Blight*). Pour ce projet, les deux maladies ont été fusionnées en une seule classe *Diseased* afin de simplifier le problème en classification binaire.

---

## Comment ça marche techniquement

L'application utilise **TensorFlow.js** pour exécuter le modèle directement côté navigateur. Concrètement :

1. L'utilisateur dépose une image
2. Le navigateur la convertit en tenseur, la redimensionne en 128x128 et normalise les pixels
3. Le modèle (chargé une fois au démarrage) fait sa prédiction localement
4. Le résultat s'affiche immédiatement

Aucune image ne quitte la machine de l'utilisateur. Aucun serveur backend n'est nécessaire — tout l'hébergement se fait sur GitHub Pages, qui sert juste des fichiers statiques.

---

## Structure du dépôt

```
leaf-diagnostic/
├── index.html              # Application web complète (HTML + CSS + JS)
├── web_model/              # Modèle converti pour TensorFlow.js
│   ├── model.json          # Architecture et métadonnées
│   └── group1-shard*.bin   # Poids du modèle
├── notebook/
│   └── MiniProjet_IA_UA3.ipynb   # Notebook d'entraînement complet
└── README.md
```

---

## Utilisation en local

### Option 1 — Cloner et servir

```bash
git clone https://github.com/tonpseudo/leaf-diagnostic.git
cd leaf-diagnostic
python -m http.server 8000
```

Puis ouvre http://localhost:8000 dans ton navigateur.

### Option 2 — Ouvrir directement la démo en ligne

Va simplement sur https://tonpseudo.github.io/leaf-diagnostic/ — pas d'installation requise.

---

## Reproduire l'entraînement

Si tu veux entraîner le modèle toi-même à partir du dataset :

1. Télécharge le [dataset Kaggle](https://www.kaggle.com/datasets/muhammadardiputra/potato-leaf-disease-dataset)
2. Ouvre `notebook/MiniProjet_IA_UA3.ipynb`
3. Adapte la variable `base_path` au chemin de ton dataset
4. Exécute les cellules dans l'ordre

Le notebook couvre l'ensemble du pipeline : exploration, réorganisation en classes binaires, entraînement, évaluation, et optimisation (comparaison Adam/SGD, étude du learning rate, EarlyStopping).

### Convertir le modèle pour TensorFlow.js

Une fois le modèle entraîné et sauvegardé en `.keras`, on le convertit avec :

```python
import tensorflow as tf
import tensorflowjs as tfjs

model = tf.keras.models.load_model("potato_model.keras")
model.export("saved_model")

# Conversion en GraphModel (plus stable que LayersModel)
# tensorflowjs_converter --input_format=tf_saved_model \
#   --output_format=tfjs_graph_model saved_model web_model
```

Le dossier `web_model/` ainsi obtenu est ce qui est servi par l'application web.

---

## Limites et pistes d'amélioration

Le modèle a été entraîné sur 900 images uniquement, ce qui reste modeste. Plusieurs pistes amélioreraient les performances :

- **Data augmentation** (rotations, flips, zooms) pour enrichir artificiellement le dataset
- **Transfer learning** depuis un modèle pré-entraîné comme MobileNet ou VGG16, qui exploite des représentations apprises sur ImageNet
- **Retour à la classification multi-classes** (Healthy, Early Blight, Late Blight) pour un diagnostic plus précis et utile en agronomie
- Pondération des classes pour compenser le ratio 1:2 entre Healthy et Diseased

---

## Technologies utilisées

- **Python 3 + TensorFlow / Keras** pour l'entraînement du modèle
- **TensorFlow.js** pour l'exécution dans le navigateur
- **HTML / CSS / JavaScript** pour l'interface utilisateur
- **GitHub Pages** pour l'hébergement gratuit

Aucune dépendance backend, aucune base de données, aucune clé API.

---

## Licence

Projet académique réalisé dans un cadre éducatif. Le dataset utilisé est sous licence Kaggle (consulter la page du dataset pour les conditions d'utilisation).

---

*Mini-projet IA · UA3 · Collège La Cité*
