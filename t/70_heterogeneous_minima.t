use strict;
use warnings;
use Test::More;
use t::Util;

my $N = 16;
my $K = 1000;
my $Ps = 99;

my $paths = [
    { b => 1000.0, l => 1.5, p => 0.0, q => 10 }
];

# Paths 1-15 are "Siren Traps" - fantastic initial latency, but low bandwidth and lossy
for (1..15) {
    push @$paths, { b => 2.0, l => 0.5, p => 0.05, q => 1 };
}

# The solvers have to balance putting some packets on the lossy traps without 
# overwhelming them, while sending the bulk to the heavy lifter.
# The landscape has many local minima due to the step functions in packet loss rounding.

my $res = run_simulation($N, $K, $Ps, $paths);

is($res->{total_alloc}, $K, "Optimizer allocates exactly K packets across 16 paths");

# To pass, the solver must recognize that it needs to use the heavy lifter for the bulk
ok($res->{allocations}[0]->{alloc} > 400, "Heavy Lifter path handles the bulk of packets (> 400)");

# It must also properly load balance the rest among the traps
my $trap_alloc_sum = 0;
for my $i (1..15) {
    $trap_alloc_sum += $res->{allocations}[$i]->{alloc};
    # No single trap should be overwhelmed
    cmp_ok($res->{allocations}[$i]->{alloc}, '<', 50, "Trap $i is not overwhelmed");
}

ok($trap_alloc_sum > 0, "Traps are still utilized efficiently");

cmp_ok($res->{total_time}, '<', 3.0, "Total time is optimal (< 3.0s)");

done_testing();
