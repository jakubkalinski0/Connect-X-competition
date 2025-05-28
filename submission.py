import numpy as np
import random
import time

# Konfiguracja (można dostosować)
AI_SEARCH_DEPTH = 4  # Głębokość przeszukiwania Minimax
CENTER_COLUMN_BIAS = 0.1  # Mały bonus za wybór środkowej kolumny, jeśli inne opcje są równe


# Funkcje pomocnicze dla gry
def drop_piece(grid, row, col, piece, config_rows):
    grid[row][col] = piece


def is_valid_location(grid, col, config_columns):
    if col < 0 or col >= config_columns:
        return False
    return grid[0][col] == 0  # Sprawdza, czy najwyższy wiersz w kolumnie jest pusty


def get_next_open_row(grid, col, config_rows):
    for r in range(config_rows - 1, -1, -1):  # Od dołu do góry
        if grid[r][col] == 0:
            return r
    return None  # Kolumna jest pełna (nie powinno się zdarzyć, jeśli is_valid_location jest używane)


def winning_move(grid, piece, config_rows, config_columns, config_inarow):
    # Sprawdzenie poziomych lokalizacji
    for c in range(config_columns - (config_inarow - 1)):
        for r in range(config_rows):
            if all(grid[r][c + i] == piece for i in range(config_inarow)):
                return True

    # Sprawdzenie pionowych lokalizacji
    for c in range(config_columns):
        for r in range(config_rows - (config_inarow - 1)):
            if all(grid[r + i][c] == piece for i in range(config_inarow)):
                return True

    # Sprawdzenie dodatnich skosów
    for c in range(config_columns - (config_inarow - 1)):
        for r in range(config_rows - (config_inarow - 1)):
            if all(grid[r + i][c + i] == piece for i in range(config_inarow)):
                return True

    # Sprawdzenie ujemnych skosów
    for c in range(config_columns - (config_inarow - 1)):
        for r in range(config_inarow - 1, config_rows):
            if all(grid[r - i][c + i] == piece for i in range(config_inarow)):
                return True
    return False


def get_valid_locations(grid, config_columns):
    valid_locations = []
    for col in range(config_columns):
        if is_valid_location(grid, col, config_columns):
            valid_locations.append(col)
    return valid_locations


def evaluate_window(window, piece, opponent_piece, config_inarow):
    score = 0
    my_pieces = np.count_nonzero(window == piece)
    opponent_pieces = np.count_nonzero(window == opponent_piece)
    empty_slots = np.count_nonzero(window == 0)

    if my_pieces == config_inarow:
        score += 100000  # Zwycięstwo
    elif my_pieces == config_inarow - 1 and empty_slots == 1:
        score += 100
    elif my_pieces == config_inarow - 2 and empty_slots == 2:
        score += 10

    if opponent_pieces == config_inarow - 1 and empty_slots == 1:
        score -= 800  # Zagrożenie przegraną jest ważniejsze niż prawie wygrana
    elif opponent_pieces == config_inarow - 2 and empty_slots == 2:
        score -= 20

    return score


def score_position(grid, piece, opponent_piece, config_rows, config_columns, config_inarow):
    score = 0

    # Ocena środkowej kolumny (promuje zajmowanie środka)
    center_array = grid[:, config_columns // 2]
    center_count = np.count_nonzero(center_array == piece)
    score += center_count * 5  # Mały bonus za każdy pionek w środku

    # Ocena poziomych
    for r in range(config_rows):
        row_array = grid[r, :]
        for c in range(config_columns - (config_inarow - 1)):
            window = row_array[c:c + config_inarow]
            score += evaluate_window(window, piece, opponent_piece, config_inarow)

    # Ocena pionowych
    for c in range(config_columns):
        col_array = grid[:, c]
        for r in range(config_rows - (config_inarow - 1)):
            window = col_array[r:r + config_inarow]
            score += evaluate_window(window, piece, opponent_piece, config_inarow)

    # Ocena dodatnich skosów
    for r in range(config_rows - (config_inarow - 1)):
        for c in range(config_columns - (config_inarow - 1)):
            window = np.array([grid[r + i][c + i] for i in range(config_inarow)])
            score += evaluate_window(window, piece, opponent_piece, config_inarow)

    # Ocena ujemnych skosów
    for r in range(config_inarow - 1, config_rows):
        for c in range(config_columns - (config_inarow - 1)):
            window = np.array([grid[r - i][c + i] for i in range(config_inarow)])
            score += evaluate_window(window, piece, opponent_piece, config_inarow)

    return score


def is_terminal_node(grid, my_piece, opponent_piece, config_rows, config_columns, config_inarow):
    return (winning_move(grid, my_piece, config_rows, config_columns, config_inarow) or
            winning_move(grid, opponent_piece, config_rows, config_columns, config_inarow) or
            len(get_valid_locations(grid, config_columns)) == 0)


def minimax(grid, depth, alpha, beta, maximizingPlayer, my_piece, opponent_piece, config_rows, config_columns,
            config_inarow, start_time, time_limit_seconds):
    valid_locations = get_valid_locations(grid, config_columns)
    is_terminal = is_terminal_node(grid, my_piece, opponent_piece, config_rows, config_columns, config_inarow)

    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(grid, my_piece, config_rows, config_columns, config_inarow):
                return (None, 10000000)  # Bardzo wysoka wartość za zwycięstwo
            elif winning_move(grid, opponent_piece, config_rows, config_columns, config_inarow):
                return (None, -10000000)  # Bardzo niska wartość za przegraną
            else:  # Remis
                return (None, 0)
        else:  # Głębokość = 0
            return (None, score_position(grid, my_piece, opponent_piece, config_rows, config_columns, config_inarow))

    if time.time() - start_time > time_limit_seconds:  # Sprawdzenie limitu czasu
        # Zwróć bieżącą najlepszą ocenę, jeśli czas się kończy
        # To uproszczenie; w praktyce można by było bardziej zaawansowanie zarządzać czasem
        return (None, score_position(grid, my_piece, opponent_piece, config_rows, config_columns, config_inarow))

    # Aby faworyzować środkowe kolumny, jeśli wyniki są podobne
    # Możemy posortować `valid_locations` tak, aby środkowe były sprawdzane jako pierwsze
    # lub dodać mały bonus do oceny środkowych kolumn (zrobione w score_position)
    # Tutaj zastosujemy prosty losowy wybór, jeśli wiele ruchów jest "najlepszych"
    # a sortowanie po środku pomoże znaleźć dobre ruchy szybciej

    center_col = config_columns // 2
    valid_locations_sorted = sorted(valid_locations, key=lambda x: abs(x - center_col))

    if maximizingPlayer:
        value = -float('inf')
        column = random.choice(valid_locations_sorted) if valid_locations_sorted else None
        for col in valid_locations_sorted:
            row = get_next_open_row(grid, col, config_rows)
            if row is None: continue  # Should not happen if is_valid_location is used

            temp_grid = grid.copy()
            drop_piece(temp_grid, row, col, my_piece, config_rows)

            # Sprawdź, czy ten ruch prowadzi do natychmiastowej wygranej
            if winning_move(temp_grid, my_piece, config_rows, config_columns, config_inarow):
                return (col, 10000000)  # Zwróć natychmiast, jeśli to wygrywający ruch

            new_score = \
            minimax(temp_grid, depth - 1, alpha, beta, False, my_piece, opponent_piece, config_rows, config_columns,
                    config_inarow, start_time, time_limit_seconds)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # Odcięcie beta
        return column, value
    else:  # Minimizing player
        value = float('inf')
        column = random.choice(valid_locations_sorted) if valid_locations_sorted else None
        for col in valid_locations_sorted:
            row = get_next_open_row(grid, col, config_rows)
            if row is None: continue

            temp_grid = grid.copy()
            drop_piece(temp_grid, row, col, opponent_piece, config_rows)

            # Sprawdź, czy ten ruch przeciwnika prowadzi do natychmiastowej wygranej dla niego
            if winning_move(temp_grid, opponent_piece, config_rows, config_columns, config_inarow):
                # Jeśli przeciwnik może wygrać, a my nie możemy wygrać w poprzednim kroku,
                # to jest to bardzo zły stan, ale to minimax go oceni jako -10000000.
                # Nie potrzebujemy tutaj specjalnego return, minimax to obsłuży.
                pass

            new_score = \
            minimax(temp_grid, depth - 1, alpha, beta, True, my_piece, opponent_piece, config_rows, config_columns,
                    config_inarow, start_time, time_limit_seconds)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break  # Odcięcie alfa
        return column, value


# Główna funkcja agenta
def act(observation, configuration):
    start_time = time.time()

    # Pobranie konfiguracji gry
    config_columns = configuration.columns
    config_rows = configuration.rows
    config_inarow = configuration.inarow

    # Ustalenie, którym graczem jest agent
    my_piece = observation.mark
    opponent_piece = 1 if my_piece == 2 else 2

    # Przekształcenie planszy na format NumPy 2D
    # Plansza w `observation.board` jest listą 1D
    board_1d = observation.board
    grid = np.array(board_1d).reshape(config_rows, config_columns)

    # Limity czasowe (60s na pierwszy ruch, potem 2s)
    # Użyjemy nieco mniej, aby mieć margines bezpieczeństwa
    if sum(1 for x in board_1d if x == 0) == config_rows * config_columns:  # Pierwszy ruch
        time_limit = 58.0
    else:
        time_limit = 1.9

        # Sprawdzenie, czy możemy wygrać w tym ruchu
    valid_locations = get_valid_locations(grid, config_columns)
    for col in valid_locations:
        temp_grid_win = grid.copy()
        row = get_next_open_row(temp_grid_win, col, config_rows)
        drop_piece(temp_grid_win, row, col, my_piece, config_rows)
        if winning_move(temp_grid_win, my_piece, config_rows, config_columns, config_inarow):
            # print(f"Agent {my_piece}: Wybieram wygrywający ruch w kolumnie {col}")
            return col

    # Sprawdzenie, czy musimy zablokować przeciwnika
    for col in valid_locations:
        temp_grid_block = grid.copy()
        row = get_next_open_row(temp_grid_block, col, config_rows)
        drop_piece(temp_grid_block, row, col, opponent_piece, config_rows)  # Symulujemy ruch przeciwnika
        if winning_move(temp_grid_block, opponent_piece, config_rows, config_columns, config_inarow):
            # print(f"Agent {my_piece}: Blokuję przeciwnika w kolumnie {col}")
            return col  # Musimy zagrać w tej kolumnie, żeby zablokować

    # Jeśli nie ma natychmiastowej wygranej ani konieczności bloku, użyj Minimax
    # Używamy globalnej głębokości, ale można by ją dynamicznie dostosowywać
    # np. w zależności od liczby wolnych pól.
    # Dla uproszczenia, stała głębokość.
    # AI_SEARCH_DEPTH jest zdefiniowane na górze

    # Dynamiczna głębokość w zależności od liczby ruchów wykonanych
    # To jest tylko przykład, można to bardziej dopracować
    moves_played = config_rows * config_columns - sum(1 for x in board_1d if x == 0)
    current_search_depth = AI_SEARCH_DEPTH
    if moves_played > (config_rows * config_columns * 0.6):  # Jeśli gra jest zaawansowana
        current_search_depth = AI_SEARCH_DEPTH + 1  # Zwiększ głębokość dla końcówki
    if moves_played > (config_rows * config_columns * 0.8):
        current_search_depth = AI_SEARCH_DEPTH + 2

    best_col, minimax_score = minimax(grid, current_search_depth, -float('inf'), float('inf'), True, my_piece,
                                      opponent_piece, config_rows, config_columns, config_inarow, start_time,
                                      time_limit - (time.time() - start_time))

    # print(f"Agent {my_piece}: Minimax wybrał kolumnę {best_col} z oceną {minimax_score}")

    if best_col is None:  # Jeśli minimax nic nie zwrócił (np. przez timeout)
        # print(f"Agent {my_piece}: Minimax nie znalazł ruchu, wybieram losowy ważny ruch.")
        valid_locations = get_valid_locations(grid, config_columns)
        if valid_locations:
            # Preferuj środkowe kolumny jako fallback
            center_col_val = config_columns // 2
            valid_locations_sorted_fallback = sorted(valid_locations, key=lambda x: abs(x - center_col_val))
            return valid_locations_sorted_fallback[0]
        else:
            return 0  # Nie powinno się zdarzyć, jeśli gra nie jest zakończona

    return best_col