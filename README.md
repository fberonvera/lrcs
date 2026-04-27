# LRCS

Julia implementation for detecting Lagrangian rotating contracting structures (LRCS) in unsteady two-dimensional velocity fields.

LRCS are materially defined regions that combine:
- finite-time contraction
- elevated accumulated intrinsic rotation, quantified by the Lagrangian-averaged vorticity deviation (LAVD)

This repository provides a practical implementation of the detection procedure described in the accompanying manuscript.

---

## Overview

The method follows this workflow:

velocity field
    ->
trajectory integration
    ->
LAVD computation
    ->
candidate region extraction
    ->
material advection of boundaries
    ->
finite-time contraction test
    ->
LRCS identification

No new diagnostics are introduced. The approach combines:
- LAVD (objective measure of intrinsic rotation)
- direct verification of finite-time contraction of material regions

---

## Files

lrcs.jl        Core implementation
lrcs_run.jl    Driver script (example workflow)
README.md      This file

---

## Requirements

- Julia (>= 1.8 recommended)
- Standard numerical libraries
- NetCDF input support (for velocity fields)

---

## Usage

Run the driver script:

julia lrcs_run.jl

The script:
- loads a velocity field
- computes LAVD
- extracts candidate regions
- advects boundaries
- identifies contracting structures

---

## Input data

The code expects:
- velocity components on a grid
- consistent spatial coordinates
- time-resolved fields

Typical units:
- velocity: m/s
- time: seconds
- coordinates: degrees (internally converted if needed)

---

## Output

The code produces:
- LAVD fields
- extracted boundaries at initial time
- advected boundaries over the time interval
- contraction diagnostics

Outputs may include NetCDF files and figures depending on configuration.

---

## Notes

- Initial localization of regions (e.g., via vorticity or streamlines) is not required but may be used to focus analysis.
- The identification of LRCS is based on objective, material criteria.
- Not all LAVD maxima yield contracting regions.
- Instantaneous streamline patterns do not reliably indicate LRCS.

---

## Reference

This code accompanies the manuscript:

F. J. Beron-Vera (2026), Lagrangian rotating contracting structures. Chaos, submitted (arXiv:)

---

## License

MIT License (or specify your preferred license)

---

## Author

F. J. Beron-Vera
University of Miami
