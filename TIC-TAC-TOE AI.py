import math


AI = 'O'
YOU = 'X'

board = ['-'] * 9

def print_board(board):
    """Print the current state of the board."""
    for i in range(0, 9, 3):
        print(board[i] + '|' + board[i+1] + '|' + board[i+2])
    print()

def check_winner(board, player):
    """Check if a player has won."""
    winning_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Horizontal
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Vertical
        [0, 4, 8], [2, 4, 6]  # Diagonal
    ]
    for combo in winning_combinations:
        if all(board[i] == player for i in combo):
            return True
    return False

def is_board_full(board):
    """Check if the board is full."""
    return all(cell != '-' for cell in board)

def minimax_alpha_beta(board, depth, alpha, beta, maximizing_player):
    """Minimax algorithm with alpha-beta pruning."""
    if check_winner(board, AI):
        return 1
    elif check_winner(board, YOU):
        return -1
    elif is_board_full(board):
        return 0

    if maximizing_player:
        max_eval = -math.inf
        for i in range(9):
            if board[i] == '-':
                board[i] = AI
                eval = minimax_alpha_beta(board, depth + 1, alpha, beta, False)
                board[i] = '-'
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
        return max_eval
    else:
        min_eval = math.inf
        for i in range(9):
            if board[i] == '-':
                board[i] = YOU
                eval = minimax_alpha_beta(board, depth + 1, alpha, beta, True)
                board[i] = '-'
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
        return min_eval

def find_best_move(board):
    """Find the best move for the AI."""
    best_move = -1
    best_eval = -math.inf
    for i in range(9):
        if board[i] == '-':
            board[i] = AI
            eval = minimax_alpha_beta(board, 0, -math.inf, math.inf, False)
            board[i] = '-'
            if eval > best_eval:
                best_eval = eval
                best_move = i
    return best_move

def get_player_move():
    """Get a valid move from the player."""
    while True:
        try:
            move = int(input("Select your move (0-8): "))
            if move >= 0 and move <= 8 and board[move] == '-':
                return move
            else:
                print("Invalid move, please choose an empty spot between 0 and 8.")
        except ValueError:
            print("Invalid input, please enter a number between 0 and 8.")

while True:
    print_board(board)

    
    move = get_player_move()
    board[move] = YOU

    if check_winner(board, YOU):
        print_board(board)
        print("You win!")
        break
    elif is_board_full(board):
        print_board(board)
        print("It's a draw!")
        break

    
    ai_move = find_best_move(board)
    board[ai_move] = AI
    print("\nAI has made its move.\n")

    if check_winner(board, AI):
        print_board(board)
        print("AI wins!")
        break
    elif is_board_full(board):
        print_board(board)
        print("It's a draw!")
        break
