use strict;
use warnings;
use Test::More;
use t::Util;

my $N = 3;
my $K = 1000;
my $Ps = 0.92;
my $paths = [
    { b => 50.0, l => 0.100, p => 0.01, q => 2 },
    { b => 50.0, l => 0.100, p => 0.01, q => 2 },
    { b => 50.0, l => 0.100, p => 0.01, q => 2 }
];

my $res = run_simulation($N, $K, $Ps, $paths);

is($res->{total_alloc}, $K, "Optimizer allocates exactly K packets");
cmp_ok($res->{total_time}, '<', 8.0, "Total time is within expected boundaries (< 8s)");
is($res->{allocations}[0]->{alloc}, 334, "Path 1 gets ~1/3 of packets");
is($res->{allocations}[1]->{alloc}, 333, "Path 2 gets ~1/3 of packets");
is($res->{allocations}[2]->{alloc}, 333, "Path 3 gets ~1/3 of packets");

done_testing();
