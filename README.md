# Relativistic ECE simulator
This package calculates the emission and absorption coefficients from suprathermal and runaway electrons. The wave polarization is approximated using cold plasma theory, while emission and absorption strengths are assumed to derive from resonance interactions with the tail electrons [[1](#harvey1993)].

A good summary of the problem and its implications can be found in section 5.2.3 of [[2](#bornatici1983)].

## Installation
At this point, as the package is incomplete, it can be installed as an editable package using pip:
```bash
pip install -e .
```
There is only one Fortran module as of yet since `f2py` is rather unreliable across platforms. While I was unable to get it working on Windows, it worked fine on my Linux machine (where Python, GCC and Meson are installed via linuxbrew) once I set up the environment variables correctly:
```bash
cd relece
export PKG_CONFIG_PATH="/home/linuxbrew/.linuxbrew/lib/pkgconfig:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="/home/linuxbrew/.linuxbrew/lib:$LD_LIBRARY_PATH"
f2py -c -m refractive_index refractive_index.f90
```

## References
1. <a id="harvey1993"></a>R. W. Harvey, M. R. O'Brien, V. V. Rozhdestvensky, T. C. Luce, M. G. McCoy, and G. D. Kerbel, [Phys. Fluids B **5**, 446 (1993)](https://dx.doi.org/10.1063/1.860530).
2. <a id="bornatici1983"></a>M. Bornatici, R. Cano, O. De Barbieri, and F. Engelmann, [Nucl. Fusion **23**, 1153 (1983)](https://dx.doi.org/10.1088/0029-5515/23/9/005).
