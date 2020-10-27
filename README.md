Projet Chatbot Industriel
----------------------------------

Le projet contient 4 dossiers : 
        - Input : Les données utilisées 
                  train.csv : Données d'entrainement
                  train_folds/train_folded : Données d'entrainement après cross_validation

        - Models : Les modèles retenus pour la prédiction
        - Notebooks : Approche LSTM sous format notebook
        - src : Code sources
                    chatbot_model.py : implémentation chatbot
                    features_extraction.py : algorithmes d'extraction de caractèristique (i.e prétraitement de texte)
                    confusion_matrix.py : Code à rajouter dans Train.py pour obtenir une matrice de confusion
                    Glove_Embed.py : Implémentation word embedding en utilisant GloVe 
                    main.py : script pour executer le chatbot
                    timetable.py : implémentation de l'ordonnancement des instructions
                    Train.py : Script d'entrainement du modèle utilisé 
                               Ce fichier contient des parser pour l'executer : 
                               python --fold <nombre_fold> --model <"nom modele"> --train_path <"chemin données"> -- extract_method <"extraction méthode">
                            
                    config.py : contenant le chemin des données d'entrainement 
                    liste.py : implémentation de méthode de stockage des instructions 
                    model_dispatcher.py : Différents algorithmes d'apprentissage utilisé tout au long du projet