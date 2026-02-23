import torch
from torch.utils.data import Dataset
import pandas as pd

from enum import IntEnum

class BoardLayer(IntEnum):
    # Pezzi Bianchi
    WHITE_P = 0; WHITE_N = 1; WHITE_B = 2; WHITE_R = 3; WHITE_Q = 4; WHITE_K = 5
    # Pezzi Neri
    BLACK_P = 6; BLACK_N = 7; BLACK_B = 8; BLACK_R = 9; BLACK_Q = 10; BLACK_K = 11
    # Metadati
    TURN = 12
    CASTLE_K = 13; CASTLE_Q = 14; CASTLE_k = 15; CASTLE_q = 16
    EN_PASSANT = 17

class ChessEncoder:
    
    def __init__(self):
        self.piece_to_index = {
            'P': BoardLayer.WHITE_P, 'N': BoardLayer.WHITE_N, 'B': BoardLayer.WHITE_B,
            'R': BoardLayer.WHITE_R, 'Q': BoardLayer.WHITE_Q, 'K': BoardLayer.WHITE_K,
            'p': BoardLayer.BLACK_P, 'n': BoardLayer.BLACK_N, 'b': BoardLayer.BLACK_B,
            'r': BoardLayer.BLACK_R, 'q': BoardLayer.BLACK_Q, 'k': BoardLayer.BLACK_K
        }
    
    def process_entry(self, fen: str, score_cp: float):
        
        x = self.encode(fen)
        y = self.normalize_score(score_cp)

        return x,y

    def encode(self, fen: str):
        """
        Converte una stringa FEN (Forsyth–Edwards Notation), che descrive una
        configurazione di una partita di scacchi, in un tensore 8x8x18.

        Formato FEN:
        - La prima sezione contiene 8 stringhe separate dal carattere '/',
        ciascuna rappresentante una traversa della scacchiera dalla 8ª alla 1ª.
        - All'interno di ogni traversa, i pezzi sono elencati da sinistra a destra
        (colonne da 'a' a 'h').
        - Le lettere maiuscole rappresentano i pezzi bianchi (P, N, B, R, Q, K),
        le lettere minuscole rappresentano i pezzi neri (p, n, b, r, q, k).
        - Le case vuote sono indicate da un numero (1–8) che specifica quante
        case vuote consecutive sono presenti.
        - Dopo la disposizione dei pezzi, sono indicati:
            * Il turno di gioco ('w' per Bianco, 'b' per Nero).
            * I diritti di arrocco ('KQkq'):
                - 'K': arrocco corto Bianco
                - 'Q': arrocco lungo Bianco
                - 'k': arrocco corto Nero
                - 'q': arrocco lungo Nero
            * La casa di en passant (es. 'e3') oppure '-' se non disponibile.
            * (Eventuali campi successivi come halfmove clock e fullmove number
            possono essere presenti ma non sono rilevanti per questa codifica.)

        Struttura del tensore di output (8x8x18):
        - Canali 0–5:   pezzi bianchi (nell’ordine: Pedone, Cavallo, Alfiere,
                        Torre, Donna, Re).
        - Canali 6–11:  pezzi neri (stesso ordine dei bianchi).
        - Canale 12:    turno di gioco (1 se tocca al Bianco, 0 se al Nero).
        - Canale 13:    diritto di arrocco corto Bianco (1 se disponibile, 0 altrimenti).
        - Canale 14:    diritto di arrocco lungo Bianco (1 se disponibile, 0 altrimenti).
        - Canale 15:    diritto di arrocco corto Nero (1 se disponibile, 0 altrimenti).
        - Canale 16:    diritto di arrocco lungo Nero (1 se disponibile, 0 altrimenti).
        - Canale 17:    casa di en passant (codificata opportunamente sulla griglia,
                        oppure tutta a zero se non disponibile).

        Parametri
        ----------
        fen : str
            Stringa FEN che descrive lo stato corrente della partita.

        Ritorna
        -------
        tensor : array-like (8, 8, 18)
            Rappresentazione tensoriale dello stato della scacchiera.
        """
        parts = fen.split(' ')
        board_part = parts[0]     
        turn = parts[1]           
        castling = parts[2]       
        en_passant = parts[3]     

        # Inizializziamo il tensore (Canali, Righe, Colonne)
        tensor = torch.zeros((18, 8, 8), dtype=torch.float32)

        # 1. Riempimento pezzi
        rows = board_part.split('/')
        for i, row in enumerate(rows):
            col = 0
            for char in row:
                if char.isdigit():
                    col += int(char)
                else:
                    p_idx = self.piece_to_index[char]
                    tensor[p_idx, i, col] = 1
                    col += 1

        # 2. Codifica Turno
        if turn == 'w':
            tensor[BoardLayer.TURN, :, :] = 1

        # 3. Codifica Arrocco
        castling_map = {
            'K': BoardLayer.CASTLE_K, 
            'Q': BoardLayer.CASTLE_Q, 
            'k': BoardLayer.CASTLE_k, 
            'q': BoardLayer.CASTLE_q
        }
        for char in castling:
            if char in castling_map:
                tensor[castling_map[char], :, :] = 1

        # 4. Codifica En Passant
        if en_passant != '-':
            ep_col = ord(en_passant[0]) - ord('a')
            ep_row = 8 - int(en_passant[1])
            # Controlliamo coordinate per sicurezza
            if 0 <= ep_row < 8 and 0 <= ep_col < 8:
                tensor[BoardLayer.EN_PASSANT, ep_row, ep_col] = 1

        return tensor 
    
    def normalize_score(self, score_cp):
        """
        Schiaccia il punteggio centipawn in un range [-1, 1].
        """
        return torch.tanh(torch.tensor(score_cp / 400.0, dtype=torch.float32))

class ChessDataset(Dataset):
    
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.encoder = ChessEncoder()
    
    # Override 
    def __len__(self):
        """
        Restitusice la lunghezza del dataset
        """
        return len(self.data)

    # Override 
    def __getitem__(self, index):
        """
        Restitusice l'i-esimo elemento del dataset in seguito all'encoding
        """
        fen = self.data.iloc[index]['FEN']

        # Gestione delle valutazioni del motore (Stockfish)
        # Nel dataset di Kaggle, le posizioni possono avere due tipi di punteggio:
        # 1. Centipedoni (es. "150"): un intero che indica il vantaggio.
        # 2. Matto forzato (es. "#3"): una stringa che indica il matto in N mosse.
        score = self.data.iloc[index]['Evaluation']

        if isinstance(score, str) and '#' in score:
            # Se il punteggio contiene '#', siamo in una situazione di matto forzato.
            # Non potendo usare il numero di mosse per la regressione, assegniamo 
            # un valore convenzionale estremo (10.000 centipedoni).
            # Il segno '-' indica se il matto è a favore del Nero.
            score = -1000 if '-' in score else 1000
        else:
            score = float(score)

        return self.encoder.process_entry(fen, score)
    
