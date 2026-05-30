use strict;
use warnings;
use Test::More;
use t::Util;

subtest "Near-Broken Link (99% Packet Loss)" => sub {
    my $N = 2;
    my $K = 100;
    my $Ps = 0.90;
    my $paths = [
        { b => 10.0, l => 0.100, p => 0.99, q => 0 },
        { b => 10.0, l => 0.100, p => 0.01, q => 0 }
    ];

    my $res = run_simulation($N, $K, $Ps, $paths);
    is($res->{total_alloc}, $K, "Optimizer allocates exactly K packets");
    cmp_ok($res->{total_time}, '<', 15.0, "Total time is within expected boundaries");
    is($res->{allocations}->[0]->{alloc}, 0, "Near-broken link receives 0 packets");
    is($res->{allocations}->[1]->{alloc}, 100, "Clean link receives all packets");
};

subtest "Unused Slow Deep-Queue Path" => sub {
    my $N = 2;
    my $K = 10;
    my $Ps = 0.95;
    my $paths = [
        { b => 100.0, l => 0.010, p => 0.00, q => 0  },
        { b => 1.0,   l => 0.500, p => 0.00, q => 50 }
    ];

    my $res = run_simulation($N, $K, $Ps, $paths);
    is($res->{total_alloc}, $K, "Optimizer allocates exactly K packets");
    cmp_ok($res->{total_time}, '<', 2.0, "Total time is within expected boundaries");
    is($res->{allocations}->[1]->{alloc}, 0, "Slow deep-queue link receives 0 packets");
};

done_testing();
