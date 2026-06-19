use strict;
use warnings;
use Test::More;
use t::Util;
use Time::HiRes qw(time);

# Define the diverse topological scenarios from our test suite
my $scenarios = {
    "1. Symmetric" => {
        N => 3, K => 1000, Ps => 92,
        paths => [
            { b => 50.0, l => 0.100, p => 0.01, q => 2 },
            { b => 50.0, l => 0.100, p => 0.01, q => 2 },
            { b => 50.0, l => 0.100, p => 0.01, q => 2 }
        ]
    },
    "2. Tortoise & Hare" => {
        N => 2, K => 1000, Ps => 95,
        paths => [
            { b => 100.0, l => 1.500, p => 0.00, q => 10 },
            { b => 10.0,  l => 0.050, p => 0.00, q => 0 }
        ]
    },
    "3. Congested Lossy" => {
        N => 4, K => 1000, Ps => 99,
        paths => [
            { b => 100.0, l => 0.050, p => 0.00, q => 5 },
            { b => 100.0, l => 0.050, p => 0.10, q => 5 },
            { b => 100.0, l => 0.050, p => 0.20, q => 5 },
            { b => 10.0,  l => 0.050, p => 0.00, q => 50 }
        ]
    },
    "4. Extreme Boundaries" => {
        N => 2, K => 1000, Ps => 99,
        paths => [
            { b => 10000.0, l => 0.010, p => 0.00, q => 0 },
            { b => 1.0,     l => 1.900, p => 0.15, q => 100 }
        ]
    },
    "5. Heterogeneous Traps" => {
        N => 16, K => 1000, Ps => 99,
        paths => [
            { b => 1000.0, l => 1.5, p => 0.0, q => 10 },
            (map { { b => 2.0, l => 0.5, p => 0.05, q => 1 } } 1..15)
        ]
    }
};

my @solvers = qw(de sa pso ga aco ts);

diag("\n" . "="x50);
diag("🏆  SOLVER LEADERBOARD  🏆");
diag("="x50);

for my $name (sort keys %$scenarios) {
    my $s = $scenarios->{$name};
    write_problem($s->{N}, $s->{K}, $s->{Ps}, $s->{paths});
    
    diag(sprintf("\n-- Scenario: %s (N=%d, K=%d) --", $name, $s->{N}, $s->{K}));
    
    my %results;
    for my $solver (@solvers) {
        my $start = time();
        my $res = execute_solver($solver, $s->{K});
        my $elapsed = time() - $start;
        $results{$solver} = { 
            sim_time => $res->{total_time}, 
            exec_time => $elapsed 
        };
    }
    
    # Sort primarily by lowest simulated time (best optimization),
    # and secondarily by fastest execution time.
    my @ranked = sort { 
        $results{$a}->{sim_time} <=> $results{$b}->{sim_time} 
        || 
        $results{$a}->{exec_time} <=> $results{$b}->{exec_time} 
    } @solvers;
    
    my $rank = 1;
    for my $solver (@ranked) {
        my $r = $results{$solver};
        my $medal = $rank == 1 ? "🥇" : ($rank == 2 ? "🥈" : ($rank == 3 ? "🥉" : "  "));
        diag(sprintf("%s #%d %-4s | Target: %8.3fs | Exec: %6.3fs", 
             $medal, $rank++, uc($solver), $r->{sim_time}, $r->{exec_time}));
    }
}

diag("\n" . "="x50 . "\n");

ok(1, "Generated solver leaderboard");
done_testing();
