use strict;
use warnings;
use Test::More;
use t::Util;

my $N = 2;
my $K = 500;
my $Ps = 0.95;
my $paths = [
    { b => 100.0, l => 0.400, p => 0.02, q => 0 },
    { b => 15.0,  l => 0.050, p => 0.01, q => 0 }
];

my $res = run_simulation($N, $K, $Ps, $paths);

is($res->{total_alloc}, $K, "Optimizer allocates exactly K packets");
cmp_ok($res->{total_time}, '<', 6.0, "Total time is within expected boundaries");
cmp_ok($res->{allocations}[0]->{alloc}, '>', 400, "Tortoise (high tput, high lat) gets majority of packets");
cmp_ok($res->{allocations}[1]->{alloc}, '<', 100, "Hare (low tput, low lat) gets minority of packets");

done_testing();
