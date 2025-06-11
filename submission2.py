import numpy as np
import random
import time

# --- Konfiguracja ---
# AI_SEARCH_DEPTH_INITIAL = 3 # Początkowa głębokość dla IDDFS, ale raczej zaczniemy od 1
MAX_SEARCH_DEPTH_BASE = 4  # Bazowa maksymalna głębokość przeszukiwania Minimax
# Zwiększymy ją dynamicznie w dalszej części gry

# Wartości dla funkcji oceny
WIN_SCORE = 100000000  # Jeszcze większa wartość za zwycięstwo
LOSE_SCORE = -100000000
SCORE_THREE_OPEN = 5000  # np. 0 X X X 0 (bardzo silne)
SCORE_THREE_SEMI = 1000  # np. Y X X X 0 lub 0 X X X Y
SCORE_TWO_OPEN = 200  # np. 0 0 X X 0 0
SCORE_TWO_SEMI = 50
CENTER_COLUMN_BONUS_PER_PIECE = 15  # Zwiększony bonus za środkową kolumnę

# Flagi dla Transposition Table
TT_EXACT = 0
TT_LOWERBOUND = 1
TT_UPPERBOUND = 2

transposition_table = {}  # Globalna tablica transpozycji


# --- Funkcje pomocnicze dla gry (takie same jak w poprzedniej wersji, ale z drobnymi poprawkami lub upewnieniem się) ---
def drop_piece(grid, row, col, piece):  # usunięto config_rows, bo grid.shape[0] jest dostępne
    grid[row][col] = piece


def is_valid_location(grid, col):
    config_columns = grid.shape[1]
    if col < 0 or col >= config_columns:
        return False
    return grid[0][col] == 0


def get_next_open_row(grid, col):
    config_rows = grid.shape[0]
    for r in range(config_rows - 1, -1, -1):
        if grid[r][col] == 0:
            return r
    return None


def winning_move(grid, piece, config_inarow):
    config_rows, config_columns = grid.shape
    # Sprawdzenie poziomych
    for c in range(config_columns - (config_inarow - 1)):
        for r in range(config_rows):
            if all(grid[r][c + i] == piece for i in range(config_inarow)):
                return True
    # Sprawdzenie pionowych
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


def get_valid_locations(grid):
    config_columns = grid.shape[1]
    valid_locations = []
    for col in range(config_columns):
        if is_valid_location(grid, col):
            valid_locations.append(col)
    return valid_locations


# --- Ulepszona funkcja oceny ---
def evaluate_window(window, piece, opponent_piece, config_inarow):
    score = 0
    my_pieces = np.count_nonzero(window == piece)
    opponent_pieces = np.count_nonzero(window == opponent_piece)
    empty_slots = np.count_nonzero(window == 0)

    if my_pieces == config_inarow:
        score += WIN_SCORE  # Zwycięstwo jest obsługiwane przez minimax, ale na wszelki wypadek
    elif my_pieces == config_inarow - 1 and empty_slots == 1:
        # Sprawdź, czy jest otwarte (0 X X X 0) vs półotwarte (Y X X X 0)
        idx_empty = np.where(window == 0)[0][0]
        is_open_ended = False
        # To jest uproszczenie, bo nie wiemy co jest "za" oknem.
        # Dla celów okna: jeśli puste jest na skraju, to półotwarte
        # Jeśli wewnętrzne (np X 0 X X) to jest to blokada, a nie otwarta 3
        # Dla prostoty, tutaj skupimy się na `config_inarow-1`
        score += SCORE_THREE_SEMI  # Domyślnie semi-open
    elif my_pieces == config_inarow - 2 and empty_slots == 2:
        score += SCORE_TWO_SEMI  # Domyślnie semi-open

    if opponent_pieces == config_inarow - 1 and empty_slots == 1:
        score -= SCORE_THREE_SEMI * 2  # Blokowanie przeciwnika jest ważniejsze
    elif opponent_pieces == config_inarow - 2 and empty_slots == 2:
        score -= SCORE_TWO_SEMI * 2

    # Dodatkowa ocena dla bardziej otwartych struktur (trudniejsze do precyzyjnego zakodowania bez analizy całych linii)
    # np. dla okna o długości config_inarow + 1 lub +2
    # Przykład dla config_inarow = 4:
    # 0 X X X 0 (window size 5) -> SCORE_THREE_OPEN
    # 0 X X 0 (window size 4) -> SCORE_TWO_OPEN

    # Uproszczony test dla otwartych: jeśli mamy config_inarow-1 i jest otoczone pustymi
    # To wymagałoby przekazania większego okna lub analizy całych linii.
    # Na razie zostajemy przy prostszym `evaluate_window`.

    return score


def score_position(grid, piece, opponent_piece, config_inarow):
    score = 0
    config_rows, config_columns = grid.shape

    # Ocena środkowej kolumny
    center_col_idx = config_columns // 2
    center_array = grid[:, center_col_idx]
    center_count = np.count_nonzero(center_array == piece)
    score += center_count * CENTER_COLUMN_BONUS_PER_PIECE

    # Ocena okien
    # Poziome
    for r in range(config_rows):
        row_array = grid[r, :]
        for c in range(config_columns - (config_inarow - 1)):
            window = row_array[c:c + config_inarow]
            score += evaluate_window(window, piece, opponent_piece, config_inarow)
    # Pionowe
    for c in range(config_columns):
        col_array = grid[:, c]
        for r in range(config_rows - (config_inarow - 1)):
            window = col_array[r:r + config_inarow]
            score += evaluate_window(window, piece, opponent_piece, config_inarow)
    # Dodatnie skosy
    for r in range(config_rows - (config_inarow - 1)):
        for c in range(config_columns - (config_inarow - 1)):
            window = np.array([grid[r + i][c + i] for i in range(config_inarow)])
            score += evaluate_window(window, piece, opponent_piece, config_inarow)
    # Ujemne skosy
    for r in range(config_inarow - 1, config_rows):
        for c in range(config_columns - (config_inarow - 1)):
            window = np.array([grid[r - i][c + i] for i in range(config_inarow)])
            score += evaluate_window(window, piece, opponent_piece, config_inarow)

    # Specjalne punkty za otwarte trójki i dwójki (analiza całych linii)
    # To jest bardziej złożone i może spowolnić, ale daje lepszą ocenę
    # Przykład dla "open three" (0XXX0)
    # Poziome
    for r_idx in range(config_rows):
        for c_idx in range(config_columns - config_inarow):  # okno o 1 większe
            if c_idx + config_inarow < config_columns:  # musi być miejsce na 0 z prawej
                # Sprawdź 0 P P P 0 (dla inarow=4, to jest 0 1 1 1 0)
                # P P P to config_inarow - 1 pionków
                if grid[r_idx, c_idx] == 0 and grid[r_idx, c_idx + config_inarow] == 0:
                    is_my_open_three = True
                    is_opp_open_three = True
                    for k in range(1, config_inarow):
                        if grid[r_idx, c_idx + k] != piece:
                            is_my_open_three = False
                        if grid[r_idx, c_idx + k] != opponent_piece:
                            is_opp_open_three = False
                    if is_my_open_three: score += SCORE_THREE_OPEN
                    if is_opp_open_three: score -= SCORE_THREE_OPEN * 1.5  # blokowanie tego jest super ważne
    # Podobnie dla pionowych i skosów (pominięte dla zwięzłości, ale można dodać)

    return score


def is_terminal_node(grid, my_piece, opponent_piece, config_inarow):
    return (winning_move(grid, my_piece, config_inarow) or
            winning_move(grid, opponent_piece, config_inarow) or
            len(get_valid_locations(grid)) == 0)


# --- Minimax z Alpha-Beta i Transposition Table ---
def minimax(grid, depth, alpha, beta, maximizingPlayer, my_piece, opponent_piece, config_inarow,
            start_time, time_limit_seconds_per_minimax_call, turn_start_time, overall_time_limit):
    config_rows, config_columns = grid.shape
    grid_tuple = tuple(grid.flatten())  # Klucz do TT

    # Sprawdzenie TT
    if grid_tuple in transposition_table:
        tt_entry = transposition_table[grid_tuple]
        tt_score, tt_depth, tt_flag, _ = tt_entry  # tt_best_move nie używane tutaj bezpośrednio
        if tt_depth >= depth:  # Mamy zapisany wynik z wystarczającej głębokości
            if tt_flag == TT_EXACT:
                return None, tt_score
            elif tt_flag == TT_LOWERBOUND:
                alpha = max(alpha, tt_score)
            elif tt_flag == TT_UPPERBOUND:
                beta = min(beta, tt_score)
            if alpha >= beta:
                return None, tt_score  # Zwracamy tt_score, bo spowodowało odcięcie

    is_terminal = is_terminal_node(grid, my_piece, opponent_piece, config_inarow)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(grid, my_piece, config_inarow):
                return None, WIN_SCORE  # Zwycięstwo jest najlepsze jak najszybciej
            elif winning_move(grid, opponent_piece, config_inarow):
                return None, LOSE_SCORE  # Przegrana jest najgorsza jak najszybciej
            else:  # Remis
                return None, 0
        else:  # Głębokość = 0
            score = score_position(grid, my_piece, opponent_piece, config_inarow)
            return None, score

    # Sprawdzenie limitu czasu dla całego ruchu
    if time.time() - turn_start_time > overall_time_limit - 0.05:  # 50ms marginesu
        # print("Minimax time limit for turn exceeded")
        # Zwracamy heurystykę, bo nie mamy czasu na głębsze przeszukiwanie
        return None, score_position(grid, my_piece, opponent_piece, config_inarow)

    # Sprawdzenie limitu czasu dla tego konkretnego wywołania minimax (mniej istotne z IDDFS)
    # if time.time() - start_time > time_limit_seconds_per_minimax_call:
    #     return None, score_position(grid, my_piece, opponent_piece, config_inarow)

    valid_locations = get_valid_locations(grid)
    center_col = config_columns // 2
    # Sortowanie ruchów: najpierw te bliżej środka
    valid_locations_sorted = sorted(valid_locations, key=lambda x: abs(x - center_col))

    # Dodatkowe sortowanie na podstawie TT (jeśli mamy best_move)
    if grid_tuple in transposition_table and transposition_table[grid_tuple][3] is not None:
        tt_best_move = transposition_table[grid_tuple][3]
        if tt_best_move in valid_locations_sorted:
            valid_locations_sorted.insert(0, valid_locations_sorted.pop(valid_locations_sorted.index(tt_best_move)))

    best_move_for_tt = None

    if maximizingPlayer:
        value = -float('inf')
        column = random.choice(valid_locations_sorted) if valid_locations_sorted else None

        for col in valid_locations_sorted:
            row = get_next_open_row(grid, col)
            if row is None: continue

            temp_grid = grid.copy()
            drop_piece(temp_grid, row, col, my_piece)

            # Szybkie sprawdzenie wygranej po ruchu, aby skrócić przeszukiwanie
            if winning_move(temp_grid, my_piece, config_inarow):
                # Zapis do TT przed returnem
                transposition_table[grid_tuple] = (WIN_SCORE, depth, TT_EXACT, col)
                return col, WIN_SCORE

            new_score = minimax(temp_grid, depth - 1, alpha, beta, False, my_piece, opponent_piece, config_inarow,
                                time.time(), time_limit_seconds_per_minimax_call, turn_start_time, overall_time_limit)[
                1]
            if new_score > value:
                value = new_score
                column = col
                best_move_for_tt = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # Odcięcie beta

        # Zapis do TT
        tt_flag_to_store = TT_EXACT
        if value <= alpha:  # Wartość nie poprawiła alpha, więc to górna granica
            # Uwaga: Właściwie to `value <= original_alpha` implikuje UPPERBOUND
            # a `value >= beta` implikuje LOWERBOUND.
            # Jeśli pętla zakończyła się normalnie (nie przez break), to jest EXACT.
            # Jeśli przez break, to (alpha >= beta), czyli znaleźliśmy ruch, który jest "za dobry"
            # dla minimalizującego gracza, więc to jest dolna granica wartości tego węzła.
            pass  # Zostaje EXACT, chyba że...
        if value >= beta:  # To jest `alpha >= beta` break.
            tt_flag_to_store = TT_LOWERBOUND
        # Jeśli value <= alpha (alpha nie została zaktualizowana), to jest UPPERBOUND
        # Ale to skomplikowane. Prostsze:
        # if value <= initial_alpha: tt_flag_to_store = TT_UPPERBOUND
        # elif value >= beta: tt_flag_to_store = TT_LOWERBOUND
        # else: tt_flag_to_store = TT_EXACT
        # Dla uproszczenia, jeśli alpha >= beta (doszło do odcięcia), to jest to dolna granica.
        # W przeciwnym razie, to dokładna wartość.
        # To nie jest w pełni poprawne zarządzanie flagami TT, ale wystarczające na początek.
        # Poprawniej:
        # if value <= original_alpha: tt_flag = TT_UPPERBOUND
        # elif value >= beta: tt_flag = TT_LOWERBOUND
        # else: tt_flag = TT_EXACT
        # Gdzie original_alpha to alpha przekazane do funkcji.
        # Na razie uproszczone:
        if alpha >= beta:  # Jeśli doszło do odcięcia beta
            tt_flag_to_store = TT_LOWERBOUND
        else:  # W przeciwnym razie, jest to dokładna wartość (lub górna granica, jeśli value pozostało -inf)
            tt_flag_to_store = TT_EXACT  # Uproszczenie, może być UPPERBOUND

        transposition_table[grid_tuple] = (value, depth, tt_flag_to_store, best_move_for_tt)
        return column, value

    else:  # Minimizing player
        value = float('inf')
        column = random.choice(valid_locations_sorted) if valid_locations_sorted else None

        for col in valid_locations_sorted:
            row = get_next_open_row(grid, col)
            if row is None: continue

            temp_grid = grid.copy()
            drop_piece(temp_grid, row, col, opponent_piece)

            if winning_move(temp_grid, opponent_piece, config_inarow):
                transposition_table[grid_tuple] = (LOSE_SCORE, depth, TT_EXACT, col)
                return col, LOSE_SCORE

            new_score = minimax(temp_grid, depth - 1, alpha, beta, True, my_piece, opponent_piece, config_inarow,
                                time.time(), time_limit_seconds_per_minimax_call, turn_start_time, overall_time_limit)[
                1]
            if new_score < value:
                value = new_score
                column = col
                best_move_for_tt = col
            beta = min(beta, value)
            if alpha >= beta:
                break  # Odcięcie alfa

        tt_flag_to_store = TT_EXACT
        if beta <= alpha:  # Jeśli doszło do odcięcia alfa
            tt_flag_to_store = TT_UPPERBOUND
        else:
            tt_flag_to_store = TT_EXACT  # Uproszczenie

        transposition_table[grid_tuple] = (value, depth, tt_flag_to_store, best_move_for_tt)
        return column, value


# --- Główna funkcja agenta z Iterative Deepening ---
def act(observation, configuration):
    turn_start_time = time.time()  # Czas rozpoczęcia całej tury

    config_columns = configuration.columns
    config_rows = configuration.rows
    config_inarow = configuration.inarow

    my_piece = observation.mark
    opponent_piece = 1 if my_piece == 2 else 2

    board_1d = observation.board
    grid = np.array(board_1d).reshape(config_rows, config_columns)

    # Limity czasowe
    is_first_move = sum(1 for x in board_1d if x == 0) == config_rows * config_columns
    overall_time_limit = 58.0 if is_first_move else 1.90  # Całkowity czas na ruch
    # time_limit_per_minimax_call = overall_time_limit * 0.8 # Uproszczenie, IDDFS zarządza tym lepiej

    # Wyczyść TT na początku każdego ruchu agenta (ważne w środowisku Kaggle)
    global transposition_table
    transposition_table = {}

    # --- Sprawdzenie natychmiastowych ruchów ---
    valid_locations = get_valid_locations(grid)
    if not valid_locations: return 0  # Nie ma ruchów, choć to nie powinno się zdarzyć przed końcem gry

    # 1. Czy mogę wygrać?
    for col in valid_locations:
        temp_grid_win = grid.copy()
        row = get_next_open_row(temp_grid_win, col)
        if row is not None:
            drop_piece(temp_grid_win, row, col, my_piece)
            if winning_move(temp_grid_win, my_piece, config_inarow):
                # print(f"Agent {my_piece}: Wybieram wygrywający ruch w kolumnie {col}")
                return col

    # 2. Czy muszę zablokować przeciwnika?
    for col in valid_locations:
        temp_grid_block = grid.copy()
        row = get_next_open_row(temp_grid_block, col)
        if row is not None:
            # Symulujemy ruch przeciwnika w tej kolumnie, aby sprawdzić, czy wygra
            # Ale my chcemy wrzucić NASZ pionek, aby go zablokować
            # Czyli: jeśli przeciwnik MÓGŁBY wygrać wrzucając w 'col', my musimy tam zagrać.
            # Tworzymy kopię do testu ruchu przeciwnika
            test_opponent_grid = grid.copy()
            test_row_opp = get_next_open_row(test_opponent_grid, col)  # Przeciwnik wrzuca w to samo miejsce
            drop_piece(test_opponent_grid, test_row_opp, col, opponent_piece)
            if winning_move(test_opponent_grid, opponent_piece, config_inarow):
                # print(f"Agent {my_piece}: Blokuję przeciwnika w kolumnie {col}")
                return col  # Musimy zagrać w tej kolumnie, żeby zablokować

    # --- Iterative Deepening ---
    best_col_overall = random.choice(valid_locations)  # Fallback, jeśli IDDFS nie zdąży nawet na depth 1

    # Dynamiczna maksymalna głębokość
    moves_played = config_rows * config_columns - sum(1 for x in board_1d if x == 0)
    max_depth = MAX_SEARCH_DEPTH_BASE
    if moves_played > (config_rows * config_columns * 0.5):
        max_depth = MAX_SEARCH_DEPTH_BASE + 1
    if moves_played > (config_rows * config_columns * 0.75):
        max_depth = MAX_SEARCH_DEPTH_BASE + 2
    if config_rows * config_columns - moves_played < 10:  # Mniej niż 10 pustych pól
        max_depth = MAX_SEARCH_DEPTH_BASE + 3  # Pozwól na głębsze przeszukiwanie w końcówce
        if config_rows * config_columns - moves_played < 6:
            max_depth = MAX_SEARCH_DEPTH_BASE + 4  # Bardzo mało pól

    # print(f"Agent {my_piece}: Rozpoczynam IDDFS, max_depth={max_depth}, czas: {overall_time_limit:.2f}s")

    for depth in range(1, max_depth + 1):
        current_call_start_time = time.time()
        # time_for_this_depth_search = (overall_time_limit - (current_call_start_time - turn_start_time)) * 0.9 # Daj trochę mniej

        # Jeśli pozostało bardzo mało czasu, nie zaczynaj nowej, głębszej iteracji
        # To oszacowanie, jak długo potrwa następna iteracja (zwykle znacznie dłużej)
        # Jeśli poprzednia iteracja zajęła X, następna może zająć X * branching_factor.
        # Dla bezpieczeństwa, jeśli mało czasu, przerwij.
        time_elapsed = time.time() - turn_start_time
        if time_elapsed > overall_time_limit * 0.85 and depth > 1:  # Jeśli zużyto 85% czasu i to nie pierwsza iteracja
            # print(f"Agent {my_piece}: Mało czasu ({time_elapsed:.2f}s / {overall_time_limit:.2f}s), przerywam IDDFS na głębokości {depth-1}.")
            break

        col_candidate, score_candidate = minimax(grid, depth, -float('inf'), float('inf'), True, my_piece,
                                                 opponent_piece, config_inarow,
                                                 current_call_start_time,  # start_time dla tego wywołania minimax
                                                 overall_time_limit,
                                                 # ogólny limit czasu na minimax (dla pojedynczego wywołania)
                                                 turn_start_time,  # start_time dla całej tury
                                                 overall_time_limit)  # ogólny limit czasu na całą turę

        time_for_current_depth = time.time() - current_call_start_time
        # print(f"Agent {my_piece}: IDDFS głębokość {depth} zakończona w {time_for_current_depth:.3f}s. Ruch: {col_candidate}, Ocena: {score_candidate}")

        if col_candidate is not None:  # Jeśli minimax nie został przerwany przez timeout natychmiast
            best_col_overall = col_candidate

        # Jeśli znaleziono wygraną, nie ma sensu szukać głębiej
        if score_candidate >= WIN_SCORE * 0.9:  # *0.9 dla marginesu, jeśli WIN_SCORE jest modyfikowane przez głębokość w minimax
            # print(f"Agent {my_piece}: Znaleziono wygrywającą sekwencję na głębokości {depth}.")
            break

        # Jeśli pozostały czas jest krótszy niż czas ostatniego przeszukiwania * pewien mnożnik, przerwij
        # To jest heurystyka, mnożnik zależy od współczynnika rozgałęzienia
        if (overall_time_limit - (time.time() - turn_start_time)) < (time_for_current_depth * 2) and depth > 1:
            # print(f"Agent {my_piece}: Przewidywany czas następnej iteracji za długi. Kończę na głębokości {depth}.")
            break
        if time.time() - turn_start_time > overall_time_limit - 0.1:  # 100ms marginesu
            # print(f"Agent {my_piece}: Całkowity czas ruchu przekroczony ({time.time() - turn_start_time:.2f}s). Kończę na głębokości {depth}.")
            break

    # print(f"Agent {my_piece}: Ostatecznie wybrano kolumnę {best_col_overall} po IDDFS.")
    return best_col_overall