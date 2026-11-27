# Hands-on: a MACE potential in LAMMPS through ML-IAP

**~15 min · Midway3 · 1 GPU (V100 or A100)**

You take a trained MACE model, convert it to an ML-IAP file, and run 100 MD steps
of liquid water (128 H₂O, 384 atoms, 300 K) with `pair_style mliap unified` + KOKKOS.

## 1. Get the files (login node)

```bash
git clone https://github.com/rcc-uchicago/mace-mliap.git
cd mace-mliap-workshop
ls
```

| File | What it is |
|---|---|
| `c-h2o.model` | Trained MACE model (PyTorch pickle); elements H, C, O; cutoff 6 Å |
| `data/water_384_300K.data` | 128 H₂O, pre-equilibrated with this model at 300 K (includes velocities) |
| `in.water.mliap` | LAMMPS input |
| `run.sbatch` | Batch job: step 1 converts the model, step 2 runs LAMMPS |

## 2. Convert MACE model to the LAMMPS MLIAP unified format

Step 1 of `run.sbatch` is the standard MACE converter:

```bash
python -m mace.cli.create_lammps_model c-h2o.model --format mliap --dtype float64
```

Arguments:

- `--format mliap` writes a pickled `LAMMPS_MLIAP_MACE` Python object, which is
  what `pair_style mliap unified` loads. Without this flag you get a TorchScript
  file for the separate `pair_style mace` (ML-MACE package).
- `--dtype float64|float32` sets the precision of the weights and the evaluation.
- `--head NAME` is only for multi-head foundation models (e.g. MACE-MH-1 needs
  `--head omol`). This model has one head.
- The ML-IAP format switches the model to **cuequivariance** (NVIDIA's
  equivariant-tensor-product GPU kernels). The kernels are picked **at
  conversion time**, so convert on a GPU node. A file converted on a login
  node still works but runs about 25× slower. That's why the conversion is
  inside the batch job.

## 3. Specify the model in the input script

The important lines of `in.water.mliap`:

```lammps
pair_style    mliap unified ${model} 0
pair_coeff    * * H O
timestep      0.0005
```

- `mliap unified <file> 0`: LAMMPS passes neighbor pairs (`rij`, `i`, `j`,
  element types) to the Python object. The object returns per-atom energies and
  per-pair forces, and LAMMPS handles the force and virial bookkeeping. With
  `0`, ML-IAP doesn't build ghost-atom neighbor lists; MACE exchanges ghost-atom
  features itself between message-passing layers.
- `pair_coeff * * H O` maps atom type 1 → H and type 2 → O by element name.
  The names must be among the model's elements.
- `timestep 0.5 fs` because the O–H bonds aren't constrained.
- The KOKKOS flags in `run.sbatch` (`-k on g 1 -sf kk -pk kokkos neigh half newton on`)
  are **required**. MACE's ML-IAP code only works with the KOKKOS version of ML-IAP.

## 4. Submit the job script

Adjust the job script `run.sbatch` for your eligible `account`.

```bash
sbatch run.sbatch
squeue --me
```

The job takes 1–1.5 min: ~15 s to convert, ~30 s of start-up (Python, CUDA,
model load), and under 10 s of MD.

## 5. Check the results

```bash
grep -A12 "^ *Step" log.float64.lammps          # thermo
grep -A8 "MPI task timing breakdown" log.float64.lammps
```

- T stays near 300 K and PotEng near −60116 eV (≈ −156.6 eV/atom; MACE
  energies include atomic reference energies). TotEng drifts only slightly,
  because the thermostat is weak.
- **Pair** (the MACE evaluation) is > 99% of the loop time.
- The first steps include warm-up. Judge speed from the `CPU` column after
  step 10: about 33 ms/step on an A100 and 46 ms/step on a V100.

## 6. Other considerations

1. **Single precision:** `sbatch --export=ALL,DTYPE=float32 run.sbatch`, then
   compare `log.float32.lammps` with `log.float64.lammps`. PotEng agrees to a
   few meV, but the speed-up is small (~12% on an A100): 384 atoms is too few to
   fill the GPU. For thousands of atoms (try the `replicate` command) float32 could be ~1.6x faster.
2. **Energy conservation:** replace the `fix nvt ...` line with `fix nve all nve`
   and watch `TotEng`; then try `timestep 0.001`.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `Incorrect args for pair coefficients` | An element name in `pair_coeff` isn't in the model, or the count doesn't match the atom types |
| `compute_forces failure` + `... 'CPU' backend` in `run-*.out` | LAMMPS ran without the KOKKOS flags, so the model ran on CPU |
| `LMP: none` | The job landed on a GPU type with no build; keep `--constraint=v100\|a100` |
| Very slow steps (~1 s/step) | The `.pt` was converted on a login node; let `run.sbatch` convert it |
| Lost atoms / T explodes | Timestep too large or a bad starting structure |
