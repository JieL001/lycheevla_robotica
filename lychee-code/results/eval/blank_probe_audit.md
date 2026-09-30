# Audit of the language probes of the blank-command arms (9/30, after an external review pointed at Table 15)

Finding: for a network trained WITHOUT commands (arm r1_blank_film, sym_r1_blank) the main record files (`s_<arm>__iid`, `__sal`) and the blank probe were evaluated with the blank command (policy tag `|blank`), but the swap probe fed the REAL command of the other member of the pair, an input the network never saw. 'Language sensitivity' (TSA normal minus TSA swapped) therefore compared two different inputs (2.3 points for R1-blank, 1.3 for the fruit-table version) instead of being exactly 0. The numbers below were computed from the files as they were BEFORE the fix; the files were then regenerated with the rule that a blank-trained checkpoint always receives the blank command (scripts/eval_select.py::command_tokens).

```
== r1_blank_film
  main (blank-mode?)           s_r1_blank_film__iid                   n=1200  TSA 30.75%  policy tag sel[r1_blank_film.pt|blank]
  blank probe                  s_r1_blank_film_blank__iid             n=1200  TSA 30.75%  policy tag sel[r1_blank_film.pt|blank]
  swap probe                   s_r1_blank_film_swap__iid              n=1200  TSA 28.42%  policy tag sel[r1_blank_film.pt|swap]
  gibberish                    s_r1_blank_film_gibberish__iid         n=1200  TSA 29.33%  policy tag sel[r1_blank_film.pt|gibberish]
  real command (normalcmd)     s_r1_blank_film_normalcmd__iid         n=1200  TSA 29.50%  policy tag sel[r1_blank_film.pt|normal]
  main vs blank probe: identical selections: True   differing: 0
  main(blank input) vs swap probe (real other command): differing selections: 521 of 1200
  real command vs swap probe: differing selections: 254 of 1200
  TSA real 29.50  swap(real other) 28.42  -> real - swap = 1.08
  TSA blank-input 30.75  swap 28.42  -> blank - swap = 2.33  (what Table 15 reports as 'lang. sens.')
== sym_r1_blank
  main (blank-mode?)           s_sym_r1_blank__iid                    n=1200  TSA 32.17%  policy tag sel[sym_r1_blank.pt|blank]
  blank probe                  s_sym_r1_blank_blank__iid              n=1200  TSA 32.17%  policy tag sel[sym_r1_blank.pt|blank]
  swap probe                   s_sym_r1_blank_swap__iid               n=1200  TSA 30.92%  policy tag sel[sym_r1_blank.pt|swap]
  gibberish                    s_sym_r1_blank_gibberish__iid          missing
  real command (normalcmd)     s_sym_r1_blank_normalcmd__iid          n=1200  TSA 26.00%  policy tag sel[sym_r1.pt|blank]
  main vs blank probe: identical selections: True   differing: 0
  main(blank input) vs swap probe (real other command): differing selections: 362 of 1200
  real command vs swap probe: differing selections: 851 of 1200
  TSA real 26.00  swap(real other) 30.92  -> real - swap = -4.92
  TSA blank-input 32.17  swap 30.92  -> blank - swap = 1.25  (what Table 15 reports as 'lang. sens.')
```
