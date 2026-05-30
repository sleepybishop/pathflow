# pathflow [![CI](https://github.com/sleepybishop/pathflow/actions/workflows/ci.yml/badge.svg)](https://github.com/sleepybishop/pathflow/actions/workflows/ci.yml)

pathflow will determine the optimum set of packets to send via a collection of paths with known characteristics:

 - Throughput (packets/s)
 - Latency
 - Loss rate
 - Packets enqueued

The problem is defined in the `problem.txt` file, in the following format:

```text
N: <number of paths>
K: <number of packets in payload>
Ps: <probability of success>
```

Followed by `N` lines in the format:

```text
<packets/s> <latency> <loss rate> <packets in queue>
```

## Build & Usage

To build the project, simply run `make`. 
This will generate the `pathflow` CLI executable and a `libpathflow.a` static library for use in your own applications.

You can verify the algorithms are working correctly by running `make test`.

## Dynamic Dashboard

You can watch the algorithm react to simulated network conditions in real-time by running the interactive ncurses dashboard:

```bash
make demo
./demo
```

The simulator creates a Markov "weather" model where links occasionally experience catastrophic drops or miraculous recoveries. The algorithm uses an EWMA filter to smooth out continuous random jitter and seamlessly redirects packet allocations on the fly. Press `q` to quit the dashboard.

## References

Implementation based on [AeroMTP][1].

[1]: https://doi.org/10.1016/j.cja.2015.05.010
