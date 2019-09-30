#README

## ORGANISATION

Il y a (4) dossiers: 

-> Data: Contient les jeux de données;

-> Inference: Contient le code source des algorithmes;

-> Results: Contient les figures des XPs;

-> Saved: Contient les modèles sauvegardés des XPs;

## Une adresse complète est composée dans l'ordre de (4) strings :

savePath + xpName + networkName + saveName

-> savePath: Contient l'adresse où sauvegarder;
-> xpName: Contient le nom l'XP;
-> networkName: Contient le nom du modèle;
-> saveName: Contient un nom plus spécifique, par exemple "LOSS" pour un graphique.



## Changements:
- Organisation des dossiers mise en place;
- Sauvegarde des modèles et figures;
- Boucle d'entraînement fonctionnelle;
- Ajout de la fonction plot_BBVI. 