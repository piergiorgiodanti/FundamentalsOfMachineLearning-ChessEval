import torch
from torch.utils.data import Dataset
import pandas as pd
from abc import ABC, abstractmethod

class BaseChessEncoder(ABC):
    """
    Interfaccia astratta per la definizione degli encoder scacchistici.
    """
    def __init__(self, scale: float = 400.0):
        self.scale = scale # Fattore di normalizzazione per la funzione Tanh

    @abstractmethod
    def process_entry(self, fen: str, score_cp: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Predisposizione della coppia input (x) e target (y) per l'addestramento."""
        pass

    @abstractmethod
    def encode(self, fen: str) -> torch.Tensor:
        """Conversione della stringa FEN in una rappresentazione tensoriale."""
        pass

    def normalize_score(self, score_cp: float) -> torch.Tensor:
        """Normalizzazione del punteggio centipedonale nell'intervallo [-1, 1]."""
        return torch.tanh(torch.tensor(score_cp / self.scale, dtype=torch.float32))

    def denormalize_score(self, model_output: torch.Tensor) -> float:
        """Conversione dell'output del modello in valore centipedonale."""
        val = torch.atanh(torch.clamp(model_output, -0.999, 0.999)).item()
        return val * self.scale

class StaticFlatEncoder(BaseChessEncoder):
    """
    Codifica a 18 canali con prospettiva fissa, sempre dal punto di vista del Bianco.
    Canali 0-11: pezzi; 12: turno; 13-16: arrocchi; 17: En Passant.
    """
    def __init__(self, scale=400.0):
        super().__init__(scale)
        self.piece_map = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5, # Pezzi bianchi (canali 0-5)
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11 # Pezzi neri (canali 6-11)
        }
        self.CH_TURN, self.CH_EP = 12, 17 # Indici canali per turno e casa di En Passant
        self.CH_CASTLE = {'K': 13, 'Q': 14, 'k': 15, 'q': 16}
        self.layers = 18

    def process_entry(self, fen, score_cp):
        return self.encode(fen), self.normalize_score(score_cp)

    def encode(self, fen):
        parts = fen.split(' ')
        tensor = torch.zeros((self.layers, 8, 8), dtype=torch.float32)

        # Mappatura della disposizione dei pezzi sulla scacchiera
        for r, row in enumerate(parts[0].split('/')):
            c = 0
            for char in row:
                if char.isdigit(): c += int(char)
                else:
                    tensor[self.piece_map[char], r, c] = 1
                    c += 1

        # Codifica del turno di gioco corrente
        if parts[1] == 'w': tensor[self.CH_TURN, :, :] = 1

        # Codifica della disponibilità dei diritti di arrocco
        for char in parts[2]:
            if char in self.CH_CASTLE: tensor[self.CH_CASTLE[char], :, :] = 1

        # Codifica della casa di En Passant, se disponibile
        if parts[3] != '-':
            c, r = ord(parts[3][0]) - ord('a'), 8 - int(parts[3][1])
            tensor[self.CH_EP, r, c] = 1
        return tensor

class PerspectiveResEncoder(BaseChessEncoder):
    """
    Codifica a 17 canali orientata al giocatore di turno.
    I canali 0-5 rappresentano sempre i pezzi del giocatore attivo.
    """
    def __init__(self, scale=400.0):
        super().__init__(scale)
        self.p_order = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}
        # Indici relativi: STM = Side To Move (giocatore attivo), OPP = Opponent (avversario)
        self.CH_STM_K, self.CH_STM_Q, self.CH_OPP_K, self.CH_OPP_Q, self.CH_EP = 12, 13, 14, 15, 16
        self.layers = 17

    def process_entry(self, fen, score_cp):
        """Inverte il segno del punteggio se il turno appartiene al Nero."""
        turn = fen.split(' ')[1]
        adjusted_score = -score_cp if turn == 'b' else score_cp
        return self.encode(fen), self.normalize_score(adjusted_score)

    def encode(self, fen):
        parts = fen.split(' ')
        turn = parts[1]
        tensor = torch.zeros((self.layers, 8, 8), dtype=torch.float32)

        # Mappatura pezzi relativa: canali 0-5 per il giocatore attivo, 6-11 per l'avversario
        for r, row in enumerate(parts[0].split('/')):
            c = 0
            for char in row:
                if char.isdigit(): c += int(char)
                else:
                    is_white = char.isupper()
                    # Identificazione della proprietà del pezzo rispetto al turno corrente
                    is_active_player = (turn == 'w' and is_white) or (turn == 'b' and not is_white)
                    p_idx = self.p_order[char.upper()]

                    channel = p_idx if is_active_player else p_idx + 6
                    tensor[channel, r, c] = 1
                    c += 1

        # Mappatura relativa dei diritti di arrocco basata sul colore del giocatore di turno
        my_k, my_q, op_k, op_q = ('K','Q','k','q') if turn == 'w' else ('k','q','K','Q')
        if my_k in parts[2]: tensor[self.CH_STM_K, :, :] = 1
        if my_q in parts[2]: tensor[self.CH_STM_Q, :, :] = 1
        if op_k in parts[2]: tensor[self.CH_OPP_K, :, :] = 1
        if op_q in parts[2]: tensor[self.CH_OPP_Q, :, :] = 1

        # Codifica En Passant
        if parts[3] != '-':
            c, r = ord(parts[3][0]) - ord('a'), 8 - int(parts[3][1])
            tensor[self.CH_EP, r, c] = 1
        return tensor

    def denormalize_score(self, model_output, turn):
        """Riconverte la valutazione realtiva al giocatore attivo (+ = vantaggio giocatore di turno) allo standard bianco."""
        val = super().denormalize_score(model_output)
        return -val if turn == 'b' else val

class ChessDataset(Dataset):
    """
    Classe per la gestione del dataset scacchistico da sorgente CSV.
    """
    def __init__(self, csv_file, encoder: BaseChessEncoder):
        self.data = pd.read_csv(csv_file)
        self.encoder = encoder
    
    def __len__(self):
        """Restituisce il numero totale di campioni nel dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """Restituisce la coppia (input, target) processata per l'indice specificato."""
        fen = self.data.iloc[index]['FEN']
        score = self.data.iloc[index]['Evaluation']

        # Normalizzazione dei punteggi relativi a situazioni di matto forzato
        if isinstance(score, str) and '#' in score:
            score = -15000 if '-' in score else 15000
        else:
            try:
                score = float(score)
            except:
                score = 0.0 # Gestione di eventuali valori non conformi
        
        # Esecuzione del processamento tramite l'encoder iniettato
        return self.encoder.process_entry(fen, score)