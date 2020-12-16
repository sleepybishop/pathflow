# pathflow

pathflow will determine the optimum set of packets to send via a collection of paths with known characteristics:

 - Throughput (packets/s)
 - Latency
 - Loss rate
 - Packets enqueued

the problem is defined in the `problem.txt` file, in the following format:

```
N: <number of paths>
K: <number of packets in payload>
Ps: <probability of success>
```

followed by `N` lines in the format 

```
<packets/s> <latency> <loss rate> <packets in queue>
```

implementation based on [AeroMTP][1]

[1]: https://doi.org/10.1016/j.cja.2015.05.010


