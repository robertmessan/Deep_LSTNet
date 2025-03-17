import torch
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt'):
        """
        patience: Nombre d'époques sans amélioration avant d'arrêter.
        verbose: Affiche les messages de suivi.
        delta: Seuil minimal d'amélioration pour considérer une amélioration.
        path: Chemin pour sauvegarder le meilleur modèle.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: compteur {self.counter} sur {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Sauvegarde le modèle lorsque la perte de validation diminue."""
        if self.verbose:
            print(f"Validation loss diminué. Sauvegarde du modèle (loss: {val_loss:.6f}).")
        torch.save(model.state_dict(), self.path)