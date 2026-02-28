import unittest 
import torch
from data_utils import StaticFlatEncoder, PerspectiveResEncoder

class TestBaseEncoderLogic(unittest.TestCase):
    """Test della logica comune definita nella classe base."""
    
    def setUp(self):
        self.encoder = StaticFlatEncoder(scale=400.0)

    def test_normalize_score(self):
        score = 100.0
        expected = 0.24491865932941437
        self.assertAlmostEqual(self.encoder.normalize_score(score).item(), expected, places=7)

    def test_denormalize_score(self):
        model_output = torch.tensor([0.24491865932941437])
        expected = 100.0
        self.assertAlmostEqual(self.encoder.denormalize_score(model_output), expected, places=2)

    def test_mate_saturation(self):
        self.assertAlmostEqual(self.encoder.normalize_score(15000.0).item(), 1.0, places=4)
        self.assertAlmostEqual(self.encoder.normalize_score(-15000.0).item(), -1.0, places=4)

    def test_zero_score_handling(self):
        self.assertEqual(self.encoder.normalize_score(0.0).item(), 0.0)


class TestStaticFlatEncoder(unittest.TestCase):

    def setUp(self):
        self.encoder = StaticFlatEncoder()

    def test_full_board_mapping_static(self):
        # P in a8, N in b7, B in c6, R in d5, Q in e4, K in f3 (Bianchi)
        # p in a1, n in b2 (Neri)
        fen = "P7/1N6/2B5/3R4/4Q3/5K2/1n6/p7 w - - 0 1"
        tensor = self.encoder.encode(fen)
        
        # Verifica Bianchi (Canali 0-5)
        expected_white = [(0,0,0), (1,1,1), (2,2,2), (3,3,3), (4,4,4), (5,5,5)]
        for p_idx, r, c in expected_white:
            self.assertEqual(tensor[p_idx, r, c], 1, f"Pezzo bianco {p_idx} fallito in {r},{c}")
            
        # Verifica Neri (Canali 6-11)
        self.assertEqual(tensor[6, 7, 0], 1) # p in a1
        self.assertEqual(tensor[7, 6, 1], 1) # n in b2

    def test_fixed_channels(self):
        """I pezzi devono rimanere nei canali originali indipendentemente dal turno."""
        fen_w = "8/8/8/8/8/8/P7/8 w - - 0 1" # Pedone bianco in a2
        fen_b = "8/8/8/8/8/8/P7/8 b - - 0 1"
        
        t_w = self.encoder.encode(fen_w)
        t_b = self.encoder.encode(fen_b)
        
        # Canale 0 è sempre il Pedone Bianco (P)
        self.assertEqual(t_w[0, 6, 0], 1)
        self.assertEqual(t_b[0, 6, 0], 1)

    def test_turn_channel_activation(self):
        """Verifica la corretta attivazione del layer del turno, 1 turno bianco; 0 turno nero"""
        fen_w = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        fen_b = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
        
        self.assertEqual(self.encoder.encode(fen_w)[self.encoder.CH_TURN, :, :].max(), 1)
        self.assertEqual(self.encoder.encode(fen_b)[self.encoder.CH_TURN, :, :].max(), 0)

    def test_castling_static_mapping(self):
        fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
        tensor = self.encoder.encode(fen)
        for char, ch_idx in self.encoder.CH_CASTLE.items():
            self.assertEqual(tensor[ch_idx, :, :].max(), 1, f"Arrocco {char} non trovato")

    def test_castling_static_mapping_on_off(self):
        fen = "r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1"
        tensor = self.encoder.encode(fen)
        self.assertEqual(tensor[self.encoder.CH_CASTLE['K'], :, :].max(), 1, f"Arrocco K non trovato")
        self.assertEqual(tensor[self.encoder.CH_CASTLE['Q'], :, :].max(), 0, f"Arrocco Q trovato")
        self.assertEqual(tensor[self.encoder.CH_CASTLE['k'], :, :].max(), 0, f"Arrocco k trovato")
        self.assertEqual(tensor[self.encoder.CH_CASTLE['q'], :, :].max(), 1, f"Arrocco q non trovato")

    def test_en_passant_static(self):
        # Casa d6 -> riga 2, colonna 3
        fen = "rnbqkbnr/ppppp1pp/8/8/4Pp2/7P/PPPP1PPR/RNBQKBN1 b Qkq e3 0 3"
        tensor = self.encoder.encode(fen)
        
        # Verifichiamo il canale 17 (CH_EP)
        self.assertEqual(tensor[self.encoder.CH_EP, 5, 4], 1, "Casa e3 non attivata nel canale 17")
        # Verifichiamo che sia l'unico bit attivo nel canale
        self.assertEqual(tensor[self.encoder.CH_EP, :, :].sum(), 1)

class TestPerspectiveResEncoder(unittest.TestCase):
    
    def setUp(self):
        self.encoder = PerspectiveResEncoder()

    def test_full_board_mapping_prespective_whites_turn(self):
        # P in a8, N in b7, B in c6, R in d5, Q in e4, K in f3 (Bianchi)
        # p in a1, n in b2 (Neri)
        fen = "P7/1N6/2B5/3R4/4Q3/5K2/1n6/p7 w - - 0 1"
        tensor = self.encoder.encode(fen)
        
        # Verifica Bianchi (Canali 0-5)
        expected_white = [(0,0,0), (1,1,1), (2,2,2), (3,3,3), (4,4,4), (5,5,5)]
        for p_idx, r, c in expected_white:
            self.assertEqual(tensor[p_idx, r, c], 1, f"Pezzo bianco {p_idx} fallito in {r},{c}")
            
        # Verifica Neri (Canali 6-11)
        self.assertEqual(tensor[6, 7, 0], 1) # p in a1
        self.assertEqual(tensor[7, 6, 1], 1) # n in b2

    def test_full_board_mapping_prespective_blacks_turn(self):
        # P in a8, N in b7, B in c6, R in d5, Q in e4, K in f3 (Bianchi)
        # p in a1, n in b2 (Neri)
        fen = "P7/1N6/2B5/3R4/4Q3/5K2/1n6/p7 b - - 0 1"
        tensor = self.encoder.encode(fen)
        
        # Verifica Bianchi (Canali 0-5)
        expected_white = [(6,0,0), (7,1,1), (8,2,2), (9,3,3), (10,4,4), (11,5,5)]
        for p_idx, r, c in expected_white:
            self.assertEqual(tensor[p_idx, r, c], 1, f"Pezzo bianco {p_idx} fallito in {r},{c}")
            
        # Verifica Neri (Canali 6-11)
        self.assertEqual(tensor[0, 7, 0], 1) # p in a1
        self.assertEqual(tensor[1, 6, 1], 1) # n in b2

    def test_piece_rotation_on_turn_change(self):
        """Stessa configurazione ma turni diversi."""
        fen_w = "P7/8/8/8/8/8/8/p7 w - - 0 1" # Bianco (P) in a8, Nero (p) in a1
        fen_b = "P7/8/8/8/8/8/8/p7 b - - 0 1"
        
        t_w = self.encoder.encode(fen_w)
        t_b = self.encoder.encode(fen_b)
        
        # Turno Bianco: P è STM (ch 0), p è OPP (ch 6)
        self.assertEqual(t_w[0, 0, 0], 1)
        self.assertEqual(t_w[6, 7, 0], 1)
        
        # Turno Nero: p è STM (ch 0), P è OPP (ch 6)
        self.assertEqual(t_b[0, 7, 0], 1)
        self.assertEqual(t_b[6, 0, 0], 1)

    def test_score_inversion_consistency(self):
        """Verifica che la valutazione sia relativa al giocatore attivo."""
        # Vantaggio Bianco +300 al turno del Bianco deve essere uguale a 
        # Vantaggio Nero -300 al turno del Nero per la rete.
        _, y_white = self.encoder.process_entry("8/8/8/8/8/8/8/8 w - - 0 1", 300.0)
        _, y_black = self.encoder.process_entry("8/8/8/8/8/8/8/8 b - - 0 1", -300.0)
        self.assertEqual(y_white.item(), y_black.item())

    def test_castling_relativity(self):
        """I diritti di arrocco devono ruotare tra canali STM (12-13) e OPP (14-15)."""
        # Solo il bianco può arroccare (K), turno al Nero
        fen = "8/8/8/8/8/8/8/8 b K - 0 1"
        tensor = self.encoder.encode(fen)
        
        # Per il Nero, l'arrocco bianco è dell'avversario (OPP_K = 14)
        self.assertEqual(tensor[self.encoder.CH_OPP_K, :, :].max(), 1)
        self.assertEqual(tensor[self.encoder.CH_STM_K, :, :].max(), 0)

        # Solo il bianco può arroccare (K), turno al Nero
        fen = "8/8/8/8/8/8/8/8 w K - 0 1"
        tensor = self.encoder.encode(fen)
        
        # Per il Nero, l'arrocco bianco è dell'avversario (OPP_K = 14)
        self.assertEqual(tensor[self.encoder.CH_STM_K, :, :].max(), 1)
        self.assertEqual(tensor[self.encoder.CH_OPP_K, :, :].max(), 0)

    def test_en_passant_boundary_mapping(self):
        # e3 -> riga 5, colonna 4
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        tensor = self.encoder.encode(fen)
        self.assertEqual(tensor[self.encoder.CH_EP, 5, 4], 1)

if __name__ == '__main__':
    unittest.main()