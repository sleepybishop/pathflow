use strict;
use warnings;
use Test::More;
use t::Util;

subtest "Extreme Low Allocation Boundary (K = 1)" => sub {
    my $N = 3;
    my $K = 1;
    my $Ps = 0.95;
    my $paths = [
        { b => 50.0, l => 0.100, p => 0.01, q => 2 },
        { b => 10.0, l => 0.050, p => 0.05, q => 5 },
        { b => 2.0,  l => 0.010, p => 0.10, q => 0 }
    ];

    my $res = run_simulation($N, $K, $Ps, $paths);
    is($res->{total_alloc}, $K, "Optimizer allocates exactly K packets");
    cmp_ok($res->{total_time}, '<', 2.0, "Total time is within expected boundaries");
};

subtest "Extreme Low Success Probability (Ps = 0.01)" => sub {
    my $N = 2;
    my $K = 500;
    my $Ps = 0.01;
    my $paths = [
        { b => 100.0, l => 0.100, p => 0.10, q => 0 },
        { b => 50.0,  l => 0.200, p => 0.20, q => 10 }
    ];

    my $res = run_simulation($N, $K, $Ps, $paths);
    is($res->{total_alloc}, $K, "Optimizer allocates exactly K packets");
};

subtest "Asymmetric Extreme Throughput Bottleneck" => sub {
    my $N = 2;
    my $K = 100;
    my $Ps = 0.95;
    my $paths = [
        { b => 100.0, l => 0.100, p => 0.01, q => 0 },
        { b => 0.1,   l => 0.010, p => 0.00, q => 0 }
    ];

    my $res = run_simulation($N, $K, $Ps, $paths);
    is($res->{total_alloc}, $K, "Optimizer allocates exactly K packets");
    cmp_ok($res->{allocations}[0]->{alloc}, '>', 95, "God link gets almost all packets");
    cmp_ok($res->{allocations}[1]->{alloc}, '<', 5, "Abysmal link gets few packets");
};

done_testing();
