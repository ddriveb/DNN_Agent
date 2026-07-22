# Pure RMSA Protocol Lock

- Primary protocol: Doherty et al. paper-parity implementation.
- C-side, MEC, split, compute delay, queue delay and deadline are absent.
- Spectrum is independent per directed arc (dual fiber).
- Strict label: Frozen Strict v1.3 pure-RMSA transfer.
- KSP-FF and FF-KSP: K=50 hops, same-hop km tie-break, highest feasible modulation, start-ascending First-Fit.
- Strict candidate pool: frozen PPO-R legal Top-30 only.

## Locked Loads

- `cost239`: [600.0]
- `nsfnet`: [250.0]
- `usnet`: [450.0, 550.0, 650.0, 900.0]
- `jpn48`: [300.0, 375.0, 475.0, 650.0]
