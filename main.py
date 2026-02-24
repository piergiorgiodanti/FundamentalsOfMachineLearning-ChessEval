import pygame
import chess
import torch
from src.data_utils import ChessEncoder
from src.models import ChessEval

# Inizzalizzazione Pygame
pygame.init()
encode = ChessEncoder()
WITDH, HEIGHT = 600, 600 # Dimensioni finestra
BOARD_SIZE = 600
SQ_SIZE = BOARD_SIZE // 8
FPS = 60
IMAGES = {}

WHITE = (235, 235, 208)
GREEN = (200, 99, 0)

def load_images():
    """Carica le immagini e le adatta alla dimensione delle case"""
    pieces = ['wp', 'wr', 'wn', 'wb', 'wk', 'wq', 'bp', 'br', 'bn', 'bb', 'bk', 'bq']
    for piece in pieces:
        IMAGES[piece] = pygame.transform.scale(pygame.image.load(f"images/{piece}.png"), (SQ_SIZE, SQ_SIZE))

def draw_board(screen):
    """Disegna la griglia 8x8 sullo schermo"""
    for row in range(8):
        for col in range(8):
            color = WHITE if ((row + col) % 2 == 0) else GREEN
            pygame.draw.rect(screen, color, (col * SQ_SIZE, row * SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_pieces(screen, board):
    """Disegna i pezzi basandosi sullo stato della board"""
    for row in range(8):
        for col in range(8):
            # pygame disegna dall'alto 0 in basso 7
            square = chess.square(col, 7 - row)
            
            # prende il pezzo in posizione (row, col) da board
            piece = board.piece_at(square)
           
            if piece is not None:
                # nome del pezzo preso
                color_prefix = 'w' if piece.color == chess.WHITE else 'b'
                piece_key = color_prefix + piece.symbol().lower()

                # disegna il pezzo sulla scacchiera
                screen.blit(IMAGES[piece_key], (col * SQ_SIZE, row * SQ_SIZE) )

def handle_move(board, current_selected_sq):
    pos = pygame.mouse.get_pos()
    col = pos[0] // SQ_SIZE
    row = pos[1] // SQ_SIZE

    clicked_sq = chess.square(col, 7 - row)

    if current_selected_sq is None: # il primo click seleziona il pezzo
        if board.piece_at(clicked_sq): # se c'è un pezzo nella casa selezionata
            return clicked_sq 
    else: # secondo click seleziona la casa di arriva
        move = chess.Move(current_selected_sq, clicked_sq) # si prova a fare la mossa

        if move in board.legal_moves: # se è legale si effettua
            board.push(move)
        
        # se è una promozione:
        elif chess.Move(current_selected_sq, clicked_sq, promotion=chess.QUEEN) in board.legal_moves:
            board.push(chess.Move(current_selected_sq, clicked_sq, promotion=chess.QUEEN))
            
        x,y = encode.process_entry(board.fen(),400)

        return None # per la prossima mossa
    
    return current_selected_sq

def main():

    # carica il modello
    device = torch.device("cpu")
    model = ChessEval().to(device)
    try:
        model.load_state_dict(torch.load("src/chess_model.pth", map_location=device))
        model.eval()
        print("Modello caricato")
    except:
        print("Modello non trovato")

    screen = pygame.display.set_mode((WITDH, HEIGHT))
    pygame.display.set_caption("chess eval")
    clock = pygame.time.Clock()
    board = chess.Board()
    load_images()
    selected_sq = None # casa selezionata per prima
    running = True
    while running:
        for event in pygame.event.get():
            
            old_fen = board.fen()

            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                selected_sq = handle_move(board, selected_sq)
            
            # Se la posizione della scacchiera è diversa si calcola il vantaggio
            if board.fen() != old_fen:
                with torch.no_grad():
                    tensor_input = encode.encode(board.fen()).unsqueeze(0).to(device)
                    output = model(tensor_input)
                    
                    score = output.item() 

                print(f"score: {score}")
                
        draw_board(screen)
        draw_pieces(screen, board)
        pygame.display.flip()
        clock.tick(FPS)        
    
    pygame.quit()

if __name__ == "__main__":
    main()