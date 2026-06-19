use strict;
use warnings;
use Test::More;
use t::Util;

my $N = 16;
my $K = 1000;
my $Ps = 99;

# Same heterogeneous traps as 70_heterogeneous_minima.t
my $paths = [
    { b => 1000.0, l => 1.5, p => 0.0, q => 10 }
];
for (1..15) {
    push @$paths, { b => 2.0, l => 0.5, p => 0.05, q => 1 };
}

# Write the problem space out once
write_problem($N, $K, $Ps, $paths);

my @solvers = qw(de sa pso ga aco ts);

for my $solver (@solvers) {
    my $res = execute_solver($solver, $K);
    
    # If the greedy initialization was correctly absorbed, the solver should
    # immediately start with a strong baseline and easily achieve optimal time.
    # If a solver wipes out the greedy candidate and relies purely on random init,
    # it is highly likely to fail to find the 2.5s optimum within 1M iterations 
    # due to the vast combinatorial search space.
    
    cmp_ok($res->{total_time}, '<', 3.0, "[$solver] Maintains greedy optimality (< 3.0s)");
    ok($res->{allocations}[0]->{alloc} > 400, "[$solver] Heavy Lifter utilized correctly");
}

done_testing();
