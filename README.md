# Dashboard de classification de races de chiens

Ce projet met en place un dashboard interactif pour explorer les données, comparer des modèles de deep learning et tester des prédictions de races de chiens sur des photos.

## En résumé

On entraîne/compare des modèles de vision par ordinateur pour reconnaître la race d’un chien sur une photo, puis on met le meilleur modèle dans une application Streamlit qui permet de tester une image et d’afficher les prédictions. Le dépôt GitHub correspond surtout à la partie démo/dashboard de prédiction, tandis que la présentation décrit aussi toute la partie expérimentation, entraînement et comparaison des modèles.

## Bibliothèques

Ce projet utilise les bibliothèques Python suivantes :

- **Streamlit :** pour l'interface web interactive du dashboard.
- **TensorFlow :** pour charger et exécuter les modèles de deep learning (ResNet50, ConvNeXt).
- **OpenCV :** pour le chargement et le prétraitement des images.
- **Pandas :** pour le traitement et l'analyse des données tabulaires (CSV).
- **NumPy :** pour les calculs numériques et la gestion des classes.
- **Altair :** pour les visualisations interactives (graphiques en barres des probabilités).
- **Scikit-learn :** pour les métriques et analyses statistiques des modèles.

## Lancement

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Fonctionnalités

- **Overview** — KPIs, distribution des classes, scores des modèles, matrices de confusion.
- **Predict** — upload ou image d'exemple, prédiction Top-K, comparaison ResNet50 vs ConvNeXt.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-5C3EE8?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=flat-square)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?style=flat-square)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=flat-square)
