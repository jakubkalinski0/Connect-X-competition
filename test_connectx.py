# test_connectx_interactive_v2.py
from kaggle_environments import make
import time

# Zaimportuj funkcję agenta z pliku submission.py
try:
    from submission import act as my_agent_function
except ImportError:
    print("BŁĄD: Nie można zaimportować agenta z submission.py. Upewnij się, że plik istnieje i jest poprawny.")
    exit()
except Exception as e:
    print(f"BŁĄD: Wystąpił inny błąd podczas importu agenta z submission.py: {e}")
    exit()


def run_match(env, agent1_func, agent2_func, agent1_name="Agent 1", agent2_name="Agent 2", render_board=False):
    """
    Rozgrywa pojedynczy mecz i zwraca wynik dla agenta1.
    Wynik: 1 (wygrana), 0.5 (remis), 0 (przegrana), -1 (błąd agenta1), -2 (błąd agenta2), -3 (błąd env.run), -4 (niespójna nagroda)
    """
    env.reset()
    try:
        game_steps = env.run([agent1_func, agent2_func])
        if not game_steps or len(game_steps) == 0:
            return -3
        final_step_state = game_steps[-1]
    except Exception as e:
        print(f"KRYTYCZNY BŁĄD podczas env.run() dla {agent1_name} vs {agent2_name}: {e}")
        return -3

    agent1_state = final_step_state[0]
    agent2_state = final_step_state[1]
    should_log_details_due_to_anomaly = False
    result_code = -4  # Domyślnie nieznany

    if agent1_state['status'] not in ["DONE", "ACTIVE"]:
        should_log_details_due_to_anomaly = True
        result_code = -1
    elif agent2_state['status'] not in ["DONE", "ACTIVE"]:
        should_log_details_due_to_anomaly = True
        result_code = -2
    else:
        reward_agent1 = agent1_state['reward']
        reward_agent2 = agent2_state['reward']

        if reward_agent1 == 1 and reward_agent2 == 0:
            result_code = 1
        elif reward_agent1 == 0 and reward_agent2 == 1:
            result_code = 0
        elif reward_agent1 == 0.5 and reward_agent2 == 0.5:
            result_code = 0.5
        elif reward_agent1 == -1 and reward_agent2 == 1 and agent1_state['status'] == 'DONE' and agent2_state[
            'status'] == 'DONE':
            should_log_details_due_to_anomaly = True
            result_code = 0
        elif reward_agent1 == 1 and reward_agent2 == -1 and agent1_state['status'] == 'DONE' and agent2_state[
            'status'] == 'DONE':
            should_log_details_due_to_anomaly = True
            result_code = 1
        elif result_code == -4:
            should_log_details_due_to_anomaly = True

    if render_board and should_log_details_due_to_anomaly:
        print(
            f"\n  Szczegóły meczu (anomalia): {agent1_name} vs {agent2_name} (Wynik dla {agent1_name}: {result_code})")
        print(env.render(mode="ansi", width=500, height=450))
    return result_code


def initialize_stats_structure(num_games_per_side):
    return {
        "total_games_played_config": num_games_per_side * 2,
        "my_agent_wins_as_p1": 0, "my_agent_losses_as_p1": 0, "my_agent_draws_as_p1": 0,
        "my_agent_errors_as_p1": 0, "opponent_errors_when_my_agent_p1": 0,
        "my_agent_wins_as_p2": 0, "my_agent_losses_as_p2": 0, "my_agent_draws_as_p2": 0,
        "my_agent_errors_as_p2": 0, "opponent_errors_when_my_agent_p2": 0,
        "env_run_errors": 0,
        "unexpected_results": 0,
        "games_as_p1_attempted": num_games_per_side,
        "games_as_p2_attempted": num_games_per_side
    }


def play_series(env, num_games, my_agent_func, my_agent_name, opponent_agent_func, opponent_agent_name,
                render_match_details_on_anomaly):
    stats = initialize_stats_structure(num_games)

    print(f"\n--- Seria: {my_agent_name} vs {opponent_agent_name} ---")
    print(f"Rozpoczynanie {num_games} gier: {my_agent_name} (P1) vs {opponent_agent_name} (P2)")
    for i in range(num_games):
        result = run_match(env, my_agent_func, opponent_agent_func, my_agent_name, opponent_agent_name,
                           render_board=render_match_details_on_anomaly)
        if result == 1:
            stats["my_agent_wins_as_p1"] += 1
        elif result == 0:
            stats["my_agent_losses_as_p1"] += 1
        elif result == 0.5:
            stats["my_agent_draws_as_p1"] += 1
        elif result == -1:
            stats["my_agent_errors_as_p1"] += 1
        elif result == -2:
            stats["opponent_errors_when_my_agent_p1"] += 1
        elif result == -3:
            stats["env_run_errors"] += 1
        elif result == -4:
            stats["unexpected_results"] += 1
        if (i + 1) % (max(1, num_games // 10)) == 0 or (i + 1) == num_games:
            print(f"  Rozegrano {i + 1}/{num_games} gier jako P1...")

    print(f"\nRozpoczynanie {num_games} gier: {opponent_agent_name} (P1) vs {my_agent_name} (P2)")
    for i in range(num_games):
        result_opponent_pov = run_match(env, opponent_agent_func, my_agent_func, opponent_agent_name, my_agent_name,
                                        render_board=render_match_details_on_anomaly)
        if result_opponent_pov == 1:
            stats["my_agent_losses_as_p2"] += 1
        elif result_opponent_pov == 0:
            stats["my_agent_wins_as_p2"] += 1
        elif result_opponent_pov == 0.5:
            stats["my_agent_draws_as_p2"] += 1
        elif result_opponent_pov == -1:
            stats["opponent_errors_when_my_agent_p2"] += 1
        elif result_opponent_pov == -2:
            stats["my_agent_errors_as_p2"] += 1
        elif result_opponent_pov == -3:
            stats["env_run_errors"] += 1
        elif result_opponent_pov == -4:
            stats["unexpected_results"] += 1
        if (i + 1) % (max(1, num_games // 10)) == 0 or (i + 1) == num_games:
            print(f"  Rozegrano {i + 1}/{num_games} gier jako P2...")
    return stats


def get_user_input():
    while True:
        try:
            num_games_str = input(
                "Podaj liczbę gier do rozegrania przeciwko każdemu przeciwnikowi (na stronę, np. 10): ")
            num_games_per_side = int(num_games_str)
            if num_games_per_side > 0:
                break
            else:
                print("Liczba gier musi być dodatnia.")
        except ValueError:
            print("Nieprawidłowa wartość. Podaj liczbę całkowitą.")

    available_opponents = {"1": "random", "2": "negamax"}
    selected_opponent_codes = []
    while True:
        print(
            "\nWybierz przeciwników do testowania (możesz wybrać wielu, wpisując numery oddzielone spacją lub przecinkiem):")
        print("  1: random")
        print("  2: negamax")
        print("  A: Obaj (random i negamax)")
        choices_str = input("Twój wybór (np. 1, 2, A): ").strip().lower()

        chosen_indices = set()
        if 'a' in choices_str:
            chosen_indices.update(['1', '2'])
        else:
            # Obsługa spacji i przecinków jako separatorów
            choices_str = choices_str.replace(',', ' ')
            chosen_indices.update(c.strip() for c in choices_str.split())

        valid_choices = True
        temp_selected_opponents = []
        for choice_idx in chosen_indices:
            if choice_idx in available_opponents:
                temp_selected_opponents.append(available_opponents[choice_idx])
            elif choice_idx:
                print(f"Nieznany wybór: {choice_idx}")
                valid_choices = False
                break

        if valid_choices and temp_selected_opponents:
            selected_opponent_codes = list(set(temp_selected_opponents))
            break
        elif not temp_selected_opponents and not ('a' in choices_str and not choices_str.replace('a', '').strip()):
            print("Nie wybrano żadnych przeciwników.")

    while True:
        render_choice = input("Czy renderować szczegóły meczów, w których wystąpią anomalie? (t/n): ").strip().lower()
        if render_choice in ['t', 'n']:
            render_match_details_on_anomaly = (render_choice == 't')
            break
        else:
            print("Nieprawidłowy wybór. Wpisz 't' lub 'n'.")

    return num_games_per_side, selected_opponent_codes, render_match_details_on_anomaly


def print_series_summary_table(stats, my_agent_name, opponent_name):
    headers = ["Rola Agenta", "Gry", "Wygrane", "Porażki", "Remisy", "% Wygranych", "Błędy Agenta",
               f"Błędy {opponent_name.split()[0]}"]
    col_widths = [max(len(h), 15) for h in headers]  # Ustalenie początkowych szerokości

    # Dane dla P1
    wins_p1 = stats['my_agent_wins_as_p1'] + stats['opponent_errors_when_my_agent_p1']
    countable_p1 = wins_p1 + stats['my_agent_losses_as_p1'] + stats['my_agent_draws_as_p1']
    win_rate_p1_str = f"{(wins_p1 / countable_p1 * 100):.1f}%" if countable_p1 > 0 else "0.0%"
    data_p1 = [
        f"{my_agent_name} (P1)", stats['games_as_p1_attempted'], wins_p1, stats['my_agent_losses_as_p1'],
        stats['my_agent_draws_as_p1'],
        win_rate_p1_str, stats['my_agent_errors_as_p1'], stats['opponent_errors_when_my_agent_p1']
    ]

    # Dane dla P2
    wins_p2 = stats['my_agent_wins_as_p2'] + stats['opponent_errors_when_my_agent_p2']
    countable_p2 = wins_p2 + stats['my_agent_losses_as_p2'] + stats['my_agent_draws_as_p2']
    win_rate_p2_str = f"{(wins_p2 / countable_p2 * 100):.1f}%" if countable_p2 > 0 else "0.0%"
    data_p2 = [
        f"{my_agent_name} (P2)", stats['games_as_p2_attempted'], wins_p2, stats['my_agent_losses_as_p2'],
        stats['my_agent_draws_as_p2'],
        win_rate_p2_str, stats['my_agent_errors_as_p2'], stats['opponent_errors_when_my_agent_p2']
    ]

    # Dane łączne dla serii
    total_wins = wins_p1 + wins_p2
    total_losses = stats['my_agent_losses_as_p1'] + stats['my_agent_losses_as_p2']
    total_draws = stats['my_agent_draws_as_p1'] + stats['my_agent_draws_as_p2']
    total_countable = total_wins + total_losses + total_draws
    total_win_rate_str = f"{(total_wins / total_countable * 100):.1f}%" if total_countable > 0 else "0.0%"
    total_my_errors = stats['my_agent_errors_as_p1'] + stats['my_agent_errors_as_p2']
    total_opponent_errors = stats['opponent_errors_when_my_agent_p1'] + stats['opponent_errors_when_my_agent_p2']
    data_total = [
        "SUMA (vs " + opponent_name + ")", stats['games_as_p1_attempted'] + stats['games_as_p2_attempted'],
        total_wins, total_losses, total_draws, total_win_rate_str, total_my_errors, total_opponent_errors
    ]

    # Aktualizacja szerokości kolumn
    all_data_rows = [data_p1, data_p2, data_total]
    for row in all_data_rows:
        for i, item in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(item)))

    # Drukowanie tabeli
    header_line = " | ".join(word.center(col_widths[i]) for i, word in enumerate(headers))
    separator_line = "-+-".join('-' * col_widths[i] for i in range(len(headers)))

    print(f"\n\n--- PODSUMOWANIE SERII: {my_agent_name} vs {opponent_name} ---")
    print(header_line)
    print(separator_line)
    for row_values in all_data_rows:
        print(" | ".join(str(val).center(col_widths[i]) for i, val in enumerate(row_values)))
    print(separator_line)
    if stats["env_run_errors"] > 0 or stats["unexpected_results"] > 0:
        print(
            f"  Dodatkowe informacje: Błędy env.run: {stats['env_run_errors']}, Niespodziewane wyniki (kod -4): {stats['unexpected_results']}")
    print("--- Koniec podsumowania serii ---")


def print_grand_total_summary(all_series_stats_dict, my_agent_name):
    if not all_series_stats_dict:
        return

    grand_total_games = 0
    grand_total_wins = 0
    grand_total_losses = 0
    grand_total_draws = 0
    grand_total_my_errors = 0
    grand_total_opponent_errors = 0
    grand_total_env_errors = 0
    grand_total_unexpected = 0

    for opponent_name, stats in all_series_stats_dict.items():
        grand_total_games += stats['total_games_played_config']
        grand_total_wins += stats['my_agent_wins_as_p1'] + stats['opponent_errors_when_my_agent_p1'] + \
                            stats['my_agent_wins_as_p2'] + stats['opponent_errors_when_my_agent_p2']
        grand_total_losses += stats['my_agent_losses_as_p1'] + stats['my_agent_losses_as_p2']
        grand_total_draws += stats['my_agent_draws_as_p1'] + stats['my_agent_draws_as_p2']
        grand_total_my_errors += stats['my_agent_errors_as_p1'] + stats['my_agent_errors_as_p2']
        grand_total_opponent_errors += stats['opponent_errors_when_my_agent_p1'] + stats[
            'opponent_errors_when_my_agent_p2']
        grand_total_env_errors += stats['env_run_errors']
        grand_total_unexpected += stats['unexpected_results']

    total_countable_for_rate = grand_total_wins + grand_total_losses + grand_total_draws
    grand_win_rate_str = f"{(grand_total_wins / total_countable_for_rate * 100):.1f}%" if total_countable_for_rate > 0 else "0.0%"

    print("\n\n=== OGÓLNE PODSUMOWANIE WSZYSTKICH SERII ===")
    print(f"Agent testowany: {my_agent_name}")
    print(f"Łączna liczba rozegranych gier (konfiguracja): {grand_total_games}")
    print(f"  Łącznie wygrane: {grand_total_wins}")
    print(f"  Łącznie porażki: {grand_total_losses}")
    print(f"  Łącznie remisy:  {grand_total_draws}")
    print(f"  Ogólny Win Rate (W/(W+L+D)): {grand_win_rate_str}")
    print(f"  Błędy {my_agent_name} (łącznie): {grand_total_my_errors}")
    print(f"  Błędy przeciwników (łącznie, liczone jako wygrane): {grand_total_opponent_errors}")
    if grand_total_env_errors > 0:
        print(f"  Błędy krytyczne env.run (łącznie): {grand_total_env_errors}")
    if grand_total_unexpected > 0:
        print(f"  Niespodziewane/Niespójne wyniki (kod -4, łącznie): {grand_total_unexpected}")
    print("==============================================")


if __name__ == "__main__":
    print("Witamy w skrypcie testującym agenta ConnectX!")

    num_games_per_side, opponents_to_test, render_match_details_on_anomaly = get_user_input()

    if not opponents_to_test:
        print("Nie wybrano żadnych przeciwników. Zakończono.")
    elif 'my_agent_function' not in globals():
        print("KRYTYCZNY BŁĄD: Agent 'my_agent_function' nie został poprawnie załadowany. Przerywam.")
    else:
        env = make("connectx", debug=False)
        my_agent_name = "Mój Agent"
        all_series_stats_dict = {}
        overall_start_time = time.time()

        print(f"\nRozpoczynanie testów. Liczba gier na stronę: {num_games_per_side}")
        print(f"Przeciwnicy: {', '.join([o.capitalize() for o in opponents_to_test])}")
        print(f"Renderowanie szczegółów meczu przy anomaliach: {'Tak' if render_match_details_on_anomaly else 'Nie'}")

        for opponent_code_name in opponents_to_test:
            opponent_display_name = opponent_code_name.capitalize()
            opponent_function = opponent_code_name

            start_series_time = time.time()
            print(f"\nTrwa testowanie przeciwko: {opponent_display_name}...")
            series_stats = play_series(env, num_games_per_side, my_agent_function, my_agent_name,
                                       opponent_function, opponent_display_name, render_match_details_on_anomaly)
            end_series_time = time.time()

            all_series_stats_dict[opponent_display_name] = series_stats
            # Drukowanie podsumowania serii zaraz po jej zakończeniu
            print_series_summary_table(series_stats, my_agent_name, opponent_display_name)
            print(f"Zakończono serię vs {opponent_display_name}. Czas: {end_series_time - start_series_time:.2f} s.")

        overall_end_time = time.time()

        # Generowanie ogólnego podsumowania na końcu
        print_grand_total_summary(all_series_stats_dict, my_agent_name)

        print(f"\nCałkowity czas wszystkich testów: {overall_end_time - overall_start_time:.2f} sekund")
        print("Zakończono testowanie.")