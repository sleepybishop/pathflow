#include "pathflow.h"
#include <math.h>
#include <ncurses.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CLAMP(x, a, b) ((x < (a)) ? (a) : ((x > (b)) ? (b) : (x)))

// Helper for generating random float between -1 and 1
float randf() { return (float)drand48() * 2.0f - 1.0f; }

#include <locale.h>

int main() {
    setlocale(LC_ALL, "");
    path_t base_path[MAX_LINKS] = {0};
    path_state_t states[MAX_LINKS] = {0};
    size_t N = 0, K = 0;
    float Ps_f = 0.0, deadline = 60.0;

    FILE *in = fopen("problem.txt", "r");
    if (!in) {
        fprintf(stderr, "Failed to open problem.txt\n");
        exit(-1);
    }

    fscanf(in, "N: %zu\n", &N);
    fscanf(in, "K: %zu\n", &K);
    fscanf(in, "Ps: %f\n", &Ps_f);
    N = (N > MAX_LINKS) ? MAX_LINKS : N;
    K = (K < 1) ? 1 : ((K > 1000) ? 1000 : K);
    Ps_f = CLAMP(Ps_f, 0.01f, 0.99f);
    size_t Ps = (size_t)roundf(Ps_f * 100.0f);

    size_t i = 0;
    for (i = 0; i < N; i++) {
        float raw_b, raw_l, raw_p;
        size_t raw_q;
        int got = fscanf(in, "%f %f %f %zu\n", &raw_b, &raw_l, &raw_p, &raw_q);
        if (got != 4)
            break;
        raw_p = CLAMP(raw_p, 0.0f, 0.99f);

        base_path[i].b = raw_b;
        base_path[i].l = raw_l;
        base_path[i].p = raw_p;
        base_path[i].q = raw_q;
    }
    fclose(in);
    if (i != N) {
        fprintf(stderr, "Invalid problem.txt format\n");
        exit(-1);
    }

    initscr();
    noecho();
    cbreak();
    nodelay(stdscr, TRUE);
    curs_set(0);
    start_color();
    use_default_colors();
    init_pair(1, COLOR_GREEN, -1);
    init_pair(2, COLOR_YELLOW, -1);
    init_pair(3, COLOR_RED, -1);
    init_pair(4, COLOR_CYAN, -1);

    srand48(time(NULL));
    float alpha = 0.1f;
    size_t tick_count = 0;

    while (1) {
        int ch = getch();
        if (ch == 'q' || ch == 'Q')
            break;

        path_t path[MAX_LINKS] = {0};
        static path_t current_path[MAX_LINKS] = {0};
        static path_t target_path[MAX_LINKS] = {0};
        static int initialized = 0;
        if (!initialized) {
            for (size_t j = 0; j < N; j++) {
                current_path[j] = base_path[j];
                target_path[j] = base_path[j];
            }
            initialized = 1;
        }

        for (size_t j = 0; j < N; j++) {
            float r = (float)drand48();
            if (r < 0.005f) { // 0.5% chance to become Stormy (catastrophic)
                target_path[j].p = CLAMP(base_path[j].p + 0.30f, 0.0f, 0.99f);
                target_path[j].l = CLAMP(base_path[j].l + 1.00f, 0.0f, 5.0f);
            } else if (r > 0.985f) { // 1.5% chance to become Sunny (perfect)
                target_path[j].p = 0.001f;
                target_path[j].l = 0.01f;
            } else if (r >
                       0.96f) { // 2.5% chance to pick a new random epoch target
                target_path[j].b = base_path[j].b *
                                   (1.0f + 0.6f * randf()); // +/- 60% bandwidth
                target_path[j].l =
                    base_path[j].l * (1.0f + 0.5f * randf()); // +/- 50% latency
                target_path[j].p =
                    base_path[j].p * (1.0f + 0.5f * randf()); // +/- 50% loss
            }

            // Drift gently towards the current epoch's weather target (3% per
            // tick)
            current_path[j].b += (target_path[j].b - current_path[j].b) * 0.03f;
            current_path[j].l += (target_path[j].l - current_path[j].l) * 0.03f;
            current_path[j].p += (target_path[j].p - current_path[j].p) * 0.03f;

            // Add a tiny bit of white noise just for visual jitter
            current_path[j].b += base_path[j].b * 0.02f * randf();
            current_path[j].l += base_path[j].l * 0.02f * randf();
            current_path[j].p += 0.005f * randf();

            // Clamp physical bounds
            current_path[j].b = CLAMP(current_path[j].b, 1.0f, 1000.0f);
            current_path[j].l = CLAMP(current_path[j].l, 0.001f, 5.0f);
            current_path[j].p = CLAMP(current_path[j].p, 0.0f, 0.99f);

            pathflow_update_state(&states[j], current_path[j].b,
                                  current_path[j].l, current_path[j].p,
                                  base_path[j].q, alpha);

            path[j].b = states[j].b_ewma;
            path[j].l = states[j].l_ewma;
            path[j].p = states[j].p_ewma;
            path[j].q = states[j].q_ewma;
        }

        path_t active_paths[MAX_LINKS] = {0};
        size_t active_N = 0;
        size_t map[MAX_LINKS] = {0};
        int dropped[MAX_LINKS] = {0};

        for (size_t j = 0; j < N; j++) {
            if (path[j].p >= 0.20f || path[j].l >= 2.0f) {
                dropped[j] = 1;
            } else {
                active_paths[active_N] = path[j];
                map[active_N] = j;
                active_N++;
            }
        }

        float total_time = 0.0f;
        if (active_N > 0) {
            total_time =
                pathflow_optimize(active_N, K, active_paths, 10000.0f, Ps);
            for (size_t j = 0; j < active_N; j++) {
                size_t orig = map[j];
                path[orig] = active_paths[j];
            }
        }

        erase();
        attron(COLOR_PAIR(4));
        mvprintw(0, 0, "=== Pathflow Dynamic EWMA Simulator ===");
        attroff(COLOR_PAIR(4));

        tick_count++;
        float elapsed = (float)tick_count * 0.1f;

        mvprintw(1, 0,
                 "Target Packets (K): %zu | Target Reliability (Ps): %zu%%", K,
                 Ps);
        mvprintw(2, 0,
                 "Est. Transfer Time: %.2fs (Deadline: %.2fs) | Elapsed: %.1fs",
                 total_time, deadline, elapsed);
        if (active_N == 0) {
            attron(COLOR_PAIR(3));
            mvprintw(2, 45, " [ERROR: ALL LINKS DROPPED]");
            attroff(COLOR_PAIR(3));
        }

        mvprintw(4, 0, "%-4s %-6s %-6s %-6s | %-12s | %s", "Link", "Tput",
                 "Lat", "Loss", "Alloc (m+x)", "Load");
        mvprintw(5, 0,
                 "-------------------------------------------------------------"
                 "-------------------");

        size_t total_alloc = 0, total_extra = 0;
        for (size_t j = 0; j < N; j++) {
            int row = 6 + j;
            mvprintw(row, 0, "[%2zu]", j);
            mvprintw(row, 5, "%6.0f", path[j].b);
            mvprintw(row, 12, "%6.3f", path[j].l);
            mvprintw(row, 19, "%5.1f%%", path[j].p * 100.0f);

            if (dropped[j]) {
                attron(COLOR_PAIR(3));
                mvprintw(row, 28, "DOWN");
                attroff(COLOR_PAIR(3));
            } else {
                size_t m = path[j].m;
                size_t x = path[j].x - path[j].m;
                total_alloc += m;
                total_extra += x;
                mvprintw(row, 28, "%4zu + %-4zu", m, x);

                int bar_start = 43;
                int max_bar = 35;
                float proportion = (float)(m + x) / (float)K;
                int blocks = (int)(proportion * max_bar);
                if (m > 0 && blocks == 0)
                    blocks = 1;

                int color = 1;
                if (path[j].p > 0.05f || path[j].l > 0.5f)
                    color = 2;

                attron(COLOR_PAIR(color));
                move(row, bar_start);
                for (int b = 0; b < max_bar; b++) {
                    if (b < blocks)
                        addch('#');
                    else
                        addch(' ');
                }
                attroff(COLOR_PAIR(color));
            }
        }

        mvprintw(6 + N + 1, 0,
                 "-------------------------------------------------------------"
                 "-------------------");
        mvprintw(6 + N + 2, 0, "TOTALS");
        mvprintw(6 + N + 2, 28, "%4zu + %-4zu", total_alloc, total_extra);

        mvprintw(LINES - 1, 0, "Press 'q' to quit.");
        refresh();

        usleep(100000);
    }

    endwin();
    return 0;
}
