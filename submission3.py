import numpy as np
import random
import time

# --- Konfiguracja ---
MAX_SEARCH_DEPTH_BASE = 4
MAX_SEARCH_DEPTH_ABSOLUTE = 16  # Maksymalna możliwa głębokość dla tablic (np. killer_moves)
# Powinna być większa niż jakakolwiek max_depth_iddfs

# Wartości dla funkcji oceny (można dostosować)
WIN_SCORE = 100000000
LOSE_SCORE = -100000000
# Bardziej zróżnicowane wartości dla różnych długości sekwencji
SCORE_N_CONFIG = {  # (liczba_pionkow_wlasnych, liczba_pustych_w_oknie_inarow) -> score
    (4, 0): WIN_SCORE,  # Zakładając inarow=4
    (3, 1): 5000,  # Silna trójka, potencjalna wygrana
    (2, 2): 200,  # Dwójka z dwoma pustymi
    (1, 3): 10,  # Jeden pionek
}
# Kary za konfiguracje przeciwnika (większe wartości absolutne)
PENALTY_N_CONFIG = {
    (4, 0): LOSE_SCORE,
    (3, 1): -15000,  # Blokowanie groźby przeciwnika jest bardzo ważne
    (2, 2): -400,
    (1, 3): -20,
}
CENTER_COLUMN_BONUS_PER_PIECE = 25  # Zwiększony bonus

# Flagi dla Transposition Table
TT_EXACT = 0
TT_LOWERBOUND = 1  # score >= value (spowodowane przez alpha cutoff)
TT_UPPERBOUND = 2  # score <= value (spowodowane przez beta cutoff lub value <= original_alpha)

transposition_table = {}
# killer_moves[ply][slot], gdzie ply to głębokość od korzenia
killer_moves = [[None, None] for _ in range(MAX_SEARCH_DEPTH_ABSOLUTE + 1)]


# --- Funkcje pomocnicze dla gry (bez zmian w stosunku do wersji z bardziej zaawansowaną ewaluacją) ---
def drop_piece(grid, row, col, piece):
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
    # Poziome
    for c in range(config_columns - (config_inarow - 1)):
        for r in range(config_rows):
            if all(grid[r][c + i] == piece for i in range(config_inarow)):
                return True
    # Pionowe
    for c in range(config_columns):
        for r in range(config_rows - (config_inarow - 1)):
            if all(grid[r + i][c] == piece for i in range(config_inarow)):
                return True
    # Dodatnie skosy
    for c in range(config_columns - (config_inarow - 1)):
        for r in range(config_rows - (config_inarow - 1)):
            if all(grid[r + i][c + i] == piece for i in range(config_inarow)):
                return True
    # Ujemne skosy
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


def evaluate_window_detailed(window, piece, opponent_piece, config_inarow):
    score = 0
    my_pieces = np.count_nonzero(window == piece)
    opponent_pieces = np.count_nonzero(window == opponent_piece)
    empty_slots = np.count_nonzero(window == 0)

    # Sprawdzenie dla gracza
    if (my_pieces, empty_slots) in SCORE_N_CONFIG:
        # Specjalna obsługa dla `config_inarow` - to jest już wygrana
        if my_pieces == config_inarow:
            return WIN_SCORE  # Zwróć od razu, to stan wygrywający
        score += SCORE_N_CONFIG[(my_pieces, empty_slots)]

    # Sprawdzenie dla przeciwnika
    if (opponent_pieces, empty_slots) in PENALTY_N_CONFIG:
        if opponent_pieces == config_inarow:
            return LOSE_SCORE  # Stan przegrywający
        score += PENALTY_N_CONFIG[(opponent_pieces, empty_slots)]

    return score


def score_position(grid, piece, opponent_piece, config_inarow):
    score = 0
    config_rows, config_columns = grid.shape

    # Ocena środkowej kolumny
    center_col_idx = config_columns // 2
    center_array = grid[:, center_col_idx]
    my_center_pieces = np.count_nonzero(center_array == piece)
    # opp_center_pieces = np.count_nonzero(center_array == opponent_piece) # Można dodać karę za pionki przeciwnika w centrum
    score += my_center_pieces * CENTER_COLUMN_BONUS_PER_PIECE

    # Ocena okien
    # Poziome
    for r in range(config_rows):
        row_array = grid[r, :]
        for c in range(config_columns - (config_inarow - 1)):
            window = row_array[c:c + config_inarow]
            eval_score = evaluate_window_detailed(window, piece, opponent_piece, config_inarow)
            if abs(eval_score) == WIN_SCORE: return eval_score  # Natychmiastowa wygrana/przegrana w oknie
            score += eval_score
    # Pionowe
    for c in range(config_columns):
        col_array = grid[:, c]
        for r in range(config_rows - (config_inarow - 1)):
            window = col_array[r:r + config_inarow]
            eval_score = evaluate_window_detailed(window, piece, opponent_piece, config_inarow)
            if abs(eval_score) == WIN_SCORE: return eval_score
            score += eval_score
    # Dodatnie skosy
    for r in range(config_rows - (config_inarow - 1)):
        for c in range(config_columns - (config_inarow - 1)):
            window = np.array([grid[r + i][c + i] for i in range(config_inarow)])
            eval_score = evaluate_window_detailed(window, piece, opponent_piece, config_inarow)
            if abs(eval_score) == WIN_SCORE: return eval_score
            score += eval_score
    # Ujemne skosy
    for r in range(config_inarow - 1, config_rows):
        for c in range(config_columns - (config_inarow - 1)):
            window = np.array([grid[r - i][c + i] for i in range(config_inarow)])
            eval_score = evaluate_window_detailed(window, piece, opponent_piece, config_inarow)
            if abs(eval_score) == WIN_SCORE: return eval_score
            score += eval_score
    return score


def is_terminal_node(grid, my_piece, opponent_piece, config_inarow):
    return (winning_move(grid, my_piece, config_inarow) or
            winning_move(grid, opponent_piece, config_inarow) or
            len(get_valid_locations(grid)) == 0)


# --- Minimax z ulepszeniami ---
def minimax(grid, depth, alpha, beta, maximizingPlayer, my_piece, opponent_piece, config_inarow,
            ply,  # Głębokość od korzenia (dla killer moves i LMR)
            turn_start_time, overall_time_limit):
    config_rows, config_columns = grid.shape
    grid_tuple = grid.tobytes()  # Szybszy klucz do TT
    original_alpha = alpha

    # --- Sprawdzenie TT ---
    tt_entry = transposition_table.get(grid_tuple)
    if tt_entry:
        tt_score, tt_depth, tt_flag, tt_best_move = tt_entry
        if tt_depth >= depth:
            if tt_flag == TT_EXACT:
                return tt_best_move, tt_score
            elif tt_flag == TT_LOWERBOUND:
                if tt_score >= beta: return tt_best_move, tt_score
                alpha = max(alpha, tt_score)
            elif tt_flag == TT_UPPERBOUND:
                if tt_score <= alpha: return tt_best_move, tt_score
                beta = min(beta, tt_score)
            # if alpha >= beta: return tt_best_move, tt_score # To jest już obsłużone

    # --- Sprawdzenie stanu terminalnego lub maksymalnej głębokości ---
    is_term = is_terminal_node(grid, my_piece, opponent_piece, config_inarow)
    if depth == 0 or is_term:
        if is_term:
            if winning_move(grid, my_piece, config_inarow):
                node_score = WIN_SCORE
            elif winning_move(grid, opponent_piece, config_inarow):
                node_score = LOSE_SCORE
            else:
                node_score = 0  # Remis
        else:  # Głębokość = 0
            node_score = score_position(grid, my_piece, opponent_piece, config_inarow)

        # Zapis do TT dla stanów liści (zawsze EXACT)
        if not tt_entry or tt_entry[1] < depth:  # Zapisz, jeśli nie ma wpisu lub jest płytszy
            transposition_table[grid_tuple] = (node_score, depth, TT_EXACT, None)
        return None, node_score

    # --- Sprawdzenie limitu czasu ---
    if time.time() - turn_start_time > overall_time_limit - 0.025:  # 25ms marginesu
        return None, score_position(grid, my_piece, opponent_piece, config_inarow)

    # --- Generowanie i sortowanie ruchów ---
    valid_locations = get_valid_locations(grid)
    ordered_moves = []

    # 1. TT Best Move
    tt_move_candidate = None
    if tt_entry:
        tt_move_candidate = tt_entry[3]
        if tt_move_candidate is not None and is_valid_location(grid, tt_move_candidate):
            ordered_moves.append(tt_move_candidate)

    # 2. Killer Moves
    if ply < MAX_SEARCH_DEPTH_ABSOLUTE:
        km1, km2 = killer_moves[ply]
        if km1 is not None and km1 != tt_move_candidate and is_valid_location(grid, km1):
            ordered_moves.append(km1)
        if km2 is not None and km2 != tt_move_candidate and km2 != km1 and is_valid_location(grid, km2):
            ordered_moves.append(km2)

    # 3. Other moves (center-biased)
    center_col = config_columns // 2
    remaining_locations = sorted(
        [loc for loc in valid_locations if loc not in ordered_moves],
        key=lambda x: abs(x - center_col)
    )
    ordered_moves.extend(remaining_locations)

    if not ordered_moves: return None, 0  # Remis, jeśli brak ruchów (choć is_terminal powinien to złapać)

    # --- Główna pętla Minimax ---
    best_move_for_node = ordered_moves[0]

    if maximizingPlayer:
        max_eval = -float('inf')
        for i, col in enumerate(ordered_moves):
            row = get_next_open_row(grid, col)
            temp_grid = grid.copy()
            drop_piece(temp_grid, row, col, my_piece)

            current_eval = 0
            # Late Move Reductions (LMR)
            # Redukcja dla ruchów, które nie są TT-move, nie są killerami, i dla głębokości >= 3
            # oraz nie są pierwszym "pozostałym" ruchem.
            # Uproszczenie: jeśli i > 0 (nie TT-move) i głębokość >=3
            # Bardziej precyzyjnie: jeśli ruch nie jest TT_move ani killerem
            is_lmr_candidate = True
            if col == tt_move_candidate: is_lmr_candidate = False
            if ply < MAX_SEARCH_DEPTH_ABSOLUTE and (col == killer_moves[ply][0] or col == killer_moves[ply][1]):
                is_lmr_candidate = False

            lmr_reduction = 0
            if depth >= 3 and is_lmr_candidate and i >= 1:  # i>=1 oznacza, że nie jest to pierwszy ruch z `ordered_moves`
                # (który mógł być TT lub killerem)
                lmr_reduction = 1  # Redukuj o 1 (np. z depth-1 do depth-2)
                if depth >= 5 and i >= 3: lmr_reduction = 2  # Większa redukcja dla późniejszych ruchów na większych głębokościach

            if lmr_reduction > 0:
                _, current_eval = minimax(temp_grid, depth - 1 - lmr_reduction, alpha, beta, False, my_piece,
                                          opponent_piece, config_inarow,
                                          ply + 1, turn_start_time, overall_time_limit)

            # Pełne przeszukiwanie, jeśli LMR dał wynik w oknie lub nie było LMR
            if lmr_reduction == 0 or current_eval > alpha:
                _, current_eval = minimax(temp_grid, depth - 1, alpha, beta, False, my_piece, opponent_piece,
                                          config_inarow,
                                          ply + 1, turn_start_time, overall_time_limit)

            if current_eval > max_eval:
                max_eval = current_eval
                best_move_for_node = col

            alpha = max(alpha, max_eval)
            if beta <= alpha:  # Beta cutoff
                if ply < MAX_SEARCH_DEPTH_ABSOLUTE and is_valid_location(grid,
                                                                         col):  # `is_valid_location` jest tu dla pewności
                    if col != killer_moves[ply][0]:
                        killer_moves[ply][1] = killer_moves[ply][0]
                        killer_moves[ply][0] = col
                break

        # Zapis do TT
        tt_flag_to_store = TT_EXACT
        if max_eval <= original_alpha:
            tt_flag_to_store = TT_UPPERBOUND
        elif max_eval >= beta:
            tt_flag_to_store = TT_LOWERBOUND

        if not tt_entry or tt_entry[1] < depth:
            transposition_table[grid_tuple] = (max_eval, depth, tt_flag_to_store, best_move_for_node)
        return best_move_for_node, max_eval

    else:  # Minimizing player
        min_eval = float('inf')
        for i, col in enumerate(ordered_moves):
            row = get_next_open_row(grid, col)
            temp_grid = grid.copy()
            drop_piece(temp_grid, row, col, opponent_piece)

            current_eval = 0
            is_lmr_candidate = True
            if col == tt_move_candidate: is_lmr_candidate = False
            if ply < MAX_SEARCH_DEPTH_ABSOLUTE and (col == killer_moves[ply][0] or col == killer_moves[ply][1]):
                is_lmr_candidate = False

            lmr_reduction = 0
            if depth >= 3 and is_lmr_candidate and i >= 1:
                lmr_reduction = 1
                if depth >= 5 and i >= 3: lmr_reduction = 2

            if lmr_reduction > 0:
                _, current_eval = minimax(temp_grid, depth - 1 - lmr_reduction, alpha, beta, True, my_piece,
                                          opponent_piece, config_inarow,
                                          ply + 1, turn_start_time, overall_time_limit)

            if lmr_reduction == 0 or current_eval < beta:
                _, current_eval = minimax(temp_grid, depth - 1, alpha, beta, True, my_piece, opponent_piece,
                                          config_inarow,
                                          ply + 1, turn_start_time, overall_time_limit)

            if current_eval < min_eval:
                min_eval = current_eval
                best_move_for_node = col

            beta = min(beta, min_eval)
            if beta <= alpha:  # Alpha cutoff
                if ply < MAX_SEARCH_DEPTH_ABSOLUTE and is_valid_location(grid, col):
                    if col != killer_moves[ply][0]:
                        killer_moves[ply][1] = killer_moves[ply][0]
                        killer_moves[ply][0] = col
                break

        tt_flag_to_store = TT_EXACT
        if min_eval >= beta:
            tt_flag_to_store = TT_LOWERBOUND  # Z perspektywy węzła MAX
        elif min_eval <= alpha:
            tt_flag_to_store = TT_UPPERBOUND

        if not tt_entry or tt_entry[1] < depth:
            transposition_table[grid_tuple] = (min_eval, depth, tt_flag_to_store, best_move_for_node)
        return best_move_for_node, min_eval


# --- Główna funkcja agenta (IDDFS) ---
def act(observation, configuration):
    turn_start_time = time.time()

    config_columns = configuration.columns
    config_rows = configuration.rows
    config_inarow = configuration.inarow

    my_piece = observation.mark
    opponent_piece = 1 if my_piece == 2 else 2

    board_1d = observation.board
    grid = np.array(board_1d).reshape(config_rows, config_columns)

    is_first_move = sum(1 for x in board_1d if x == 0) == config_rows * config_columns
    overall_time_limit = 58.5 if is_first_move else 1.90  # Zwiększony czas na pierwszy ruch

    global transposition_table, killer_moves
    transposition_table = {}
    killer_moves = [[None, None] for _ in range(MAX_SEARCH_DEPTH_ABSOLUTE + 1)]

    # --- Sprawdzenie natychmiastowych ruchów ---
    valid_locations = get_valid_locations(grid)
    if not valid_locations: return 0

    # 1. Czy mogę wygrać?
    for col_win in valid_locations:
        temp_grid_win = grid.copy()
        row_win = get_next_open_row(temp_grid_win, col_win)
        if row_win is not None:
            drop_piece(temp_grid_win, row_win, col_win, my_piece)
            if winning_move(temp_grid_win, my_piece, config_inarow):
                return col_win

    # 2. Czy muszę zablokować przeciwnika?
    #    Jeśli jest więcej niż 1 sposób na blok, minimax to rozważy.
    #    Jeśli jest tylko 1, to jest to ruch przymusowy.
    #    Sprawdźmy, czy jest tylko JEDEN ruch blokujący (bo inaczej minimax).
    #    W praktyce: jeśli JEST ruch blokujący, wykonaj go. Minimax by go wybrał, ale wolniej.
    for col_block in valid_locations:
        # Sprawdź, czy przeciwnik wygrywa, jeśli zagra w col_block
        test_opponent_grid = grid.copy()
        test_row_opp = get_next_open_row(test_opponent_grid, col_block)
        if test_row_opp is not None:
            drop_piece(test_opponent_grid, test_row_opp, col_block, opponent_piece)
            if winning_move(test_opponent_grid, opponent_piece, config_inarow):
                return col_block  # Musimy zagrać w tej kolumnie, żeby zablokować

    # --- Iterative Deepening z Aspiration Windows ---
    best_col_overall = random.choice(valid_locations)
    last_score_from_iddfs = 0

    # Dynamiczna maksymalna głębokość
    num_empty_cells = sum(1 for x in board_1d if x == 0)
    max_depth_iddfs = MAX_SEARCH_DEPTH_BASE
    if num_empty_cells < config_rows * config_columns * 0.60: max_depth_iddfs += 1
    if num_empty_cells < config_rows * config_columns * 0.40: max_depth_iddfs += 1
    if num_empty_cells < 15: max_depth_iddfs += 1  # Końcówka gry
    if num_empty_cells < 8: max_depth_iddfs += 2  # Bardzo blisko końca
    if max_depth_iddfs >= MAX_SEARCH_DEPTH_ABSOLUTE: max_depth_iddfs = MAX_SEARCH_DEPTH_ABSOLUTE - 1

    time_spent_on_last_full_iter = 0.01
    # print(f"Agent {my_piece} | Max depth for IDDFS: {max_depth_iddfs}")

    for current_search_depth in range(1, max_depth_iddfs + 1):
        iter_start_time = time.time()

        time_elapsed_total = iter_start_time - turn_start_time
        remaining_time_for_turn = overall_time_limit - time_elapsed_total

        # Bardziej ostrożne zarządzanie czasem
        # Współczynnik rozgałęzienia może być ~5-7. Czas ~ BF^głębokość.
        # Jeśli poprzednia iteracja zajęła T, następna może zająć T * BF.
        # Dla bezpieczeństwa, jeśli mało czasu, przerwij.
        estimated_time_for_this_iter = time_spent_on_last_full_iter * 4.5  # Ostrożny średni BF
        if current_search_depth > 1 and remaining_time_for_turn < estimated_time_for_this_iter * 1.2:  # *1.2 margines
            # print(f"Agent {my_piece}: Mało czasu ({remaining_time_for_turn:.2f}s), przewidywany ({estimated_time_for_this_iter:.2f}s). Przerywam IDDFS na głębokości {current_search_depth-1}.")
            break
        if remaining_time_for_turn < 0.05:  # Absolutne minimum, żeby zdążyć zwrócić wynik
            break

        alpha_aspir, beta_aspir = -float('inf'), float('inf')
        # SCORE_N_CONFIG[(2,2)] to około 200. Dajmy margines trochę większy.
        aspiration_window_margin = SCORE_N_CONFIG.get((2, 2), 200) * 2.5

        if current_search_depth > 2:  # Zastosuj Aspiration od głębokości 3 (dla depth=1,2 wyniki mogą być niestabilne)
            alpha_aspir = last_score_from_iddfs - aspiration_window_margin
            beta_aspir = last_score_from_iddfs + aspiration_window_margin

        col_candidate, score_candidate = minimax(grid, current_search_depth, alpha_aspir, beta_aspir, True, my_piece,
                                                 opponent_piece, config_inarow, 0,  # ply = 0 dla korzenia
                                                 turn_start_time, overall_time_limit)

        # Jeśli wynik wypadł poza Aspiration Window, przeszukaj ponownie z pełnym oknem
        # (lub odpowiednio zawężonym, jeśli znamy kierunek "porażki" aspiracji)
        if score_candidate <= alpha_aspir or score_candidate >= beta_aspir:
            # print(f"Agent {my_piece}: IDDFS głębokość {current_search_depth}: Aspiration fail ({score_candidate} vs [{alpha_aspir}, {beta_aspir}]). Re-searching.")
            alpha_re, beta_re = -float('inf'), float('inf')
            # Można by zawęzić bardziej inteligentnie:
            # if score_candidate <= alpha_aspir: beta_re = score_candidate (lub alpha_aspir)
            # if score_candidate >= beta_aspir: alpha_re = score_candidate (lub beta_aspir)
            col_candidate, score_candidate = minimax(grid, current_search_depth, alpha_re, beta_re, True, my_piece,
                                                     opponent_piece, config_inarow, 0,
                                                     turn_start_time, overall_time_limit)

        current_iter_duration = time.time() - iter_start_time

        # Aktualizuj tylko jeśli minimax nie został przerwany przez główny timeout
        if time.time() - turn_start_time < overall_time_limit - 0.01:
            if col_candidate is not None:
                best_col_overall = col_candidate
                last_score_from_iddfs = score_candidate
            time_spent_on_last_full_iter = current_iter_duration  # Zapisz czas tej iteracji, jeśli się zakończyła poprawnie
        else:  # Minimax prawdopodobnie przerwał z powodu timeoutu, nie ufaj wynikom w pełni
            # print(f"Agent {my_piece}: IDDFS głębokość {current_search_depth} przerwana przez timeout. Używam wyniku z głębokości {current_search_depth-1}.")
            break

            # print(f"Agent {my_piece}: IDDFS głębokość {current_search_depth} w {current_iter_duration:.3f}s. Ruch: {best_col_overall}, Ocena: {last_score_from_iddfs}")

        if score_candidate >= WIN_SCORE * 0.9:  # *0.9 dla pewności, bo WIN_SCORE może być modyfikowane w minimaxie
            # print(f"Agent {my_piece}: Znaleziono wygrywającą sekwencję na głębokości {current_search_depth}.")
            break
        # Nie przerywaj na pewnej przegranej, bo może istnieć dłuższa droga do przegranej (co jest lepsze)
        # lub przeciwnik może popełnić błąd.

    # print(f"Agent {my_piece}: Ostatecznie wybrano kolumnę {best_col_overall} po IDDFS (ostatnia głębokość: {current_search_depth-1 if current_iter_duration > remaining_time_for_turn else current_search_depth}). Ocena: {last_score_from_iddfs}")
    return best_col_overall