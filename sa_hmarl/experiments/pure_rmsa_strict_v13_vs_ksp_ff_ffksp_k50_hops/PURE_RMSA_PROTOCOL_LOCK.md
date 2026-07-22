# Pure RMSA Protocol Lock

- Traffic and physical core: Doherty 2025 paper-parity implementation.
- Requests contain only src, dst, bitrate, arrival and holding time.
- Interarrival: exponential with mean `10/load`; bitrate: integer uniform 25-100 Gbps.
- Holding: exponential mean 10, redrawn unless `0 < holding < 20` (observed mean about 6.9).
- 100 slots per directed arc, 12.5 GHz slots, one guard slot.
- Modulations in order: BPSK/QPSK/8QAM/16QAM; reach 10000/2500/1250/625 km.
- K=50 paths, hops then km; 10 blocks per path/mod, `start_asc`.
- Strict is always labelled `Frozen Strict v1.3 pure-RMSA transfer`.
- No C-side, MEC, deadline, compute delay, server allocation or queue state exists.

## Locked Loads

- `cost239`: [600.0]
- `nsfnet`: [250.0]
- `usnet`: [450.0, 550.0, 650.0, 900.0]
- `jpn48`: [300.0, 375.0, 475.0, 650.0]

## First Ten COST239 Requests

```json
[
  {
    "req_id": 0,
    "src": 2,
    "dst": 8,
    "arrival": 0.003599323555362411,
    "interarrival": 0.003599323555362411,
    "holding": 1.1001814174231692,
    "bitrate_gbps": 38
  },
  {
    "req_id": 1,
    "src": 8,
    "dst": 2,
    "arrival": 0.023201168335715897,
    "interarrival": 0.019601844780353487,
    "holding": 13.850632610707233,
    "bitrate_gbps": 61
  },
  {
    "req_id": 2,
    "src": 10,
    "dst": 9,
    "arrival": 0.024778748324961873,
    "interarrival": 0.001577579989245976,
    "holding": 0.15925528367396757,
    "bitrate_gbps": 26
  },
  {
    "req_id": 3,
    "src": 0,
    "dst": 10,
    "arrival": 0.06935804400588025,
    "interarrival": 0.044579295680918375,
    "holding": 2.0473654804707917,
    "bitrate_gbps": 47
  },
  {
    "req_id": 4,
    "src": 4,
    "dst": 9,
    "arrival": 0.07322000511011585,
    "interarrival": 0.0038619611042356033,
    "holding": 3.1744569571089953,
    "bitrate_gbps": 37
  },
  {
    "req_id": 5,
    "src": 4,
    "dst": 0,
    "arrival": 0.07973059802020385,
    "interarrival": 0.0065105929100880006,
    "holding": 5.3211920189500255,
    "bitrate_gbps": 75
  },
  {
    "req_id": 6,
    "src": 5,
    "dst": 0,
    "arrival": 0.08947473237315026,
    "interarrival": 0.00974413435294641,
    "holding": 13.089708745504913,
    "bitrate_gbps": 47
  },
  {
    "req_id": 7,
    "src": 0,
    "dst": 6,
    "arrival": 0.09012834856052274,
    "interarrival": 0.0006536161873724744,
    "holding": 11.994998906753015,
    "bitrate_gbps": 65
  },
  {
    "req_id": 8,
    "src": 3,
    "dst": 0,
    "arrival": 0.09574335716522615,
    "interarrival": 0.005615008604703417,
    "holding": 0.4131353508671934,
    "bitrate_gbps": 64
  },
  {
    "req_id": 9,
    "src": 0,
    "dst": 8,
    "arrival": 0.11860694163501351,
    "interarrival": 0.022863584469787354,
    "holding": 18.921795581458053,
    "bitrate_gbps": 43
  }
]
```
