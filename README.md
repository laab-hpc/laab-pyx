# LAAB-PyX

*Version*: v2026.1

LAAB-PyX benchmarks the linear-algebra awareness of PyTorch, TensorFlow, and
JAX. See [Benchmark Suite](BENCHMARK_SUITE.md) for the tests and evaluation
method.

Run the following commands from the repository root.

## 1. Prepare the virtual environment

[JUBE 2](https://apps.fz-juelich.de/jsc/jube/jube2/docu/) must be installed.

The environment recipe is
`venvs/jube/prepare_venv_kebnekaise.xml`. Before running it, update only these
values if required:

- `venv_tag`: output environment name.
- `modules`: modules that provide Python and the required system packages.
- `requirements`: any additional Python packages.

Create and activate the environment:

```bash
(cd venvs/jube && jube run prepare_venv_kebnekaise.xml)
source venvs/pyx-kebnekaise/activate.sh
python -m pip install -r mkdocs-requirements.txt
```

## 2. Run the benchmarks

Submit all five tests for PyTorch, TensorFlow, and JAX:

```bash
(cd jube/cpu-mt-v2026/kebnekaise:zen4/PyTorch-2.1.2-foss-2023a && bash run.sh)
(cd jube/cpu-mt-v2026/kebnekaise:zen4/TensorFlow-2.15.1-foss-2023a && bash run.sh)
(cd jube/cpu-mt-v2026/kebnekaise:zen4/jax-0.4.25-gfbf-2023a && bash run.sh)
```

Each `run.sh` submits tests `t1` through `t5` for 1 and 4 CPUs. Wait for the
Slurm jobs to finish before processing the results.

## 3. Collect profiles and open the dashboard

```bash
pyx-collect-profiles jube/cpu-mt-v2026/kebnekaise:zen4
pyx-dashboard jube/cpu-mt-v2026/kebnekaise:zen4 --port 8010
```

Open <http://127.0.0.1:8010/>.

## 4. Generate and view Markdown reports

```bash
./prepare_docs.sh
mkdocs serve -f mkdocs.yaml
```

Open <http://127.0.0.1:8000/>.

`prepare_docs.sh` collects the profiles, regenerates the Markdown reports,
and performs a strict MkDocs build.

Reports are organized as:

```text
docs/
└── framework/
    └── version-toolchain/
        └── system/
            └── test-id/
                └── cb-id--ncpus.md
```

## Citation

```bibtex
@inproceedings{sankaran2022benchmarking,
  title     = {Benchmarking the Linear Algebra Awareness of TensorFlow and PyTorch},
  author    = {Sankaran, Aravind and Alashti, Navid Akbari and
               Psarras, Christos and Bientinesi, Paolo},
  booktitle = {Proceedings of the 2022 IEEE International Parallel and
               Distributed Processing Symposium Workshops (IPDPSW)},
  pages     = {924--933},
  year      = {2022},
  address   = {Lyon, France},
  publisher = {IEEE},
  doi       = {10.1109/IPDPSW55747.2022.00150}
}
```

## Acknowledgements

Financial support was provided by the Deutsche Forschungsgemeinschaft (DFG,
German Research Foundation) through Grant IRTG 2379 and by the BMBF project
“01-1H1-6013 AP6-NRW Anwenderunterstützung SiVeGCS”.

Computing resources were provided by the High Performance Computing Center
North (HPC2N), Sweden, and the Jülich Supercomputing Centre (JSC), Germany.

## License

LAAB-PyX is licensed under the GNU Affero General Public License v3.0. See
[LICENSE](LICENSE) for the complete terms.
