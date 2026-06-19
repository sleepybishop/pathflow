package t::Util;
use strict;
use warnings;
use Exporter 'import';

our @EXPORT = qw(run_simulation write_problem execute_solver);

sub write_problem {
    my ($N, $K, $Ps, $paths) = @_;
    open(my $fh, '>', "problem.txt") or die "Cannot write problem.txt: $!";
    print $fh "N: $N\nK: $K\nPs: $Ps\n";
    for my $path (@$paths) {
        print $fh "$path->{b} $path->{l} $path->{p} $path->{q}\n";
    }
    close($fh);
}

sub execute_solver {
    my ($algorithm, $K) = @_;
    my $cmd = "./pathflow";
    
    local $ENV{PATHFLOW_SOLVER} = $algorithm if $algorithm;
    my $output = `$cmd 2>&1`;
    my $exit_code = $?;
    
    if ($exit_code != 0) {
        die "Error running $cmd! Exit code: $exit_code\nOutput:\n$output\n";
    }
    
    my $total_alloc = 0;
    my $total_time = 0;
    my @allocations;
    
    for my $line (split /\n/, $output) {
        if ($line =~ /estimated transfer time:\s*([\d\.]+)/) {
            $total_time = $1;
        } elsif ($line =~ /m\[\s*(\d+)\]:\s+(\d+)\s+\+\s+(\d+)\s+\[\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\|\s*([\d\.]+)\s*\]\s*->\s*([\d\.]+)/) {
            push @allocations, { id => $1, alloc => $2, tput => $4, lat => $5, loss => $6, time => $7 };
        } elsif ($line =~ /m\[xx\]:\s+(\d+)\s+(\d+)/) {
            $total_alloc = $1;
        }
    }
    
    return {
        total_time  => $total_time,
        total_alloc => $total_alloc,
        allocations => \@allocations
    };
}

sub run_simulation {
    my ($N, $K, $Ps, $paths) = @_;
    write_problem($N, $K, $Ps, $paths);
    my $res = execute_solver('hybrid', $K);
    return $res;
}

1;
