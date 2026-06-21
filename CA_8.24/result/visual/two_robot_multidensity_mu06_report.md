# Two-Robot Placement Test, mu=0.6

- Map: 32 x 37; exits: [(16, 36), (17, 36)]; fires: [(15, 15), (15, 16), (16, 15), (16, 16)]
- Pedestrians: [40, 70, 100, 120, 140]; seeds per condition: 30; paired seeds are reused across layouts.
- Exit rule: shared two-cell physical exit, interval 2 steps/person.
- Robot repulsion: cutoff 5.0 cells (2.0 m), amplitude 0.25; conflict friction base mu=0.6.

## Overall ranking across N

- Aligned-A1 (15,32)+(18,32): mean gain 28.49 steps; mean friction reduction 578.7; improved/worsened 138/10 paired runs.
- Wide-A1 (14,32)+(19,32): mean gain 19.44 steps; mean friction reduction 489.9; improved/worsened 121/27 paired runs.
- Staggered-S1 (15,30)+(18,32): mean gain 17.83 steps; mean friction reduction 496.5; improved/worsened 121/28 paired runs.
- Staggered-S2 (18,28)+(15,32): mean gain 16.45 steps; mean friction reduction 455.9; improved/worsened 117/29 paired runs.
- Staggered-S0 (18,30)+(15,32): mean gain 16.40 steps; mean friction reduction 478.7; improved/worsened 116/32 paired runs.
- Aligned-A0 (15,30)+(18,30): mean gain 2.35 steps; mean friction reduction 350.8; improved/worsened 83/63 paired runs.

## Aligned A0 vs Staggered S0

- N=40: S0 faster than A0 by 6.87 steps on average, 95% CI [2.75, 10.98], better seeds 21/30.
- N=70: S0 faster than A0 by 12.97 steps on average, 95% CI [7.79, 18.14], better seeds 24/30.
- N=100: S0 faster than A0 by 8.37 steps on average, 95% CI [2.23, 14.50], better seeds 21/30.
- N=120: S0 faster than A0 by 19.03 steps on average, 95% CI [10.71, 27.36], better seeds 23/30.
- N=140: S0 faster than A0 by 23.03 steps on average, 95% CI [14.93, 31.13], better seeds 24/30.

## Conclusion

Best robust layout in this test is Aligned-A1 (15,32)+(18,32). The requested aligned placement (15,30)+(18,30) is not the best because it is too far upstream to reduce the effective friction at the exit candidate cells; staggered layouts with one robot around column 32 reduce door-side conflict friction more directly.
