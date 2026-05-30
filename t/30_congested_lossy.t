use strict;
use warnings;
use Test::More;
use t::Util;

subtest "High Loss Congested Path" => sub {
    my $N = 2;
    my $K = 800;
    my $Ps = 0.98;
    my $paths = [
        { b => 200.0, l => 0.050, p => 0.15, q => 40 },
        { b => 30.0,  l => 0.020, p => 0.00, q => 0  }
    ];

    my $res = run_simulation($N, $K, $Ps, $paths);
    is($res->{total_alloc}, $K, "Optimizer allocates exactly K packets");
cmp_ok($res->{total_time}, '<', 15.0, "Total time is within expected boundaries");
cmp_ok($res->{allocations}[0]->{alloc}, '>', 100, "Clean path gets reasonable share despite queue");
cmp_ok($res->{allocations}[1]->{alloc}, '>', 100, "Lossy path gets reasonable share despite loss");
};

subtest "Diverse Multi-Link Environment" => sub {
    my $N = 4;
    my $K = 1000;
    my $Ps = 0.90;
    my $paths = [
        { b => 80.0,  l => 0.050, p => 0.01, q => 5  },
        { b => 40.0,  l => 0.120, p => 0.04, q => 10 },
        { b => 10.0,  l => 0.200, p => 0.08, q => 2  },
        { b => 5.0,   l => 0.350, p => 0.15, q => 0  }
    ];

    my $res = run_simulation($N, $K, $Ps, $paths);
    is($res->{total_alloc}, $K, "Optimizer allocates exactly K packets");
};

done_testing();
