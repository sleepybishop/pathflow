use strict;
use warnings;
use Test::More;
use t::Util;

my $N = 16;
my $K = 1000;
my $Ps = 0.95;
my $paths = [
    { b => 150.0, l => 0.020, p => 0.005, q => 0 },
    { b => 120.0, l => 0.035, p => 0.010, q => 5 },
    
    { b => 5.0,   l => 0.450, p => 0.250, q => 2 },
    { b => 2.0,   l => 0.600, p => 0.350, q => 0 },
    
    { b => 80.0,  l => 0.080, p => 0.020, q => 10 },
    { b => 70.0,  l => 0.090, p => 0.025, q => 8  },
    { b => 60.0,  l => 0.110, p => 0.030, q => 6  },
    { b => 55.0,  l => 0.120, p => 0.035, q => 4  },
    { b => 50.0,  l => 0.130, p => 0.040, q => 12 },
    { b => 45.0,  l => 0.150, p => 0.045, q => 3  },
    { b => 40.0,  l => 0.160, p => 0.050, q => 15 },
    { b => 35.0,  l => 0.180, p => 0.060, q => 5  },
    { b => 30.0,  l => 0.200, p => 0.070, q => 2  },
    { b => 25.0,  l => 0.220, p => 0.080, q => 10 },
    { b => 20.0,  l => 0.250, p => 0.100, q => 4  },
    { b => 15.0,  l => 0.300, p => 0.120, q => 1  }
];

my $res = run_simulation($N, $K, $Ps, $paths);

is($res->{total_alloc}, $K, "Optimizer allocates exactly K packets");
cmp_ok($res->{total_time}, '<', 30.0, "Total time is within expected boundaries");
cmp_ok($res->{allocations}[0]->{alloc}, '>', 0, "Links receive packets");

done_testing();
