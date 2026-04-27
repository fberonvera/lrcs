# LRCS

Julia implementation for detecting Lagrangian rotating contracting structures (LRCS) in unsteady two-dimensional velocity fields.

LRCS are materially defined regions that combine:
- finite-time contraction
- elevated accumulated intrinsic rotation, quantified by the Lagrangian-averaged vorticity deviation (LAVD)

This repository provides an implementation of the detection procedure described in:

F. J. Beron-Vera (2026), "Lagrangian rotating contracting structures", Chaos (submitted).

---

## Origin of the code

This Julia implementation is an adaptation, with assistance from ChatGPT, of MATLAB codes originally developed by F. J. Beron-Vera. The numerical methodology, detection strategy, and scientific design originate from those MATLAB implementations. The present code translates that workflow into Julia while preserving its structure and intent.

---

## Overview

The method separates candidate identification from coherence verification.

LAVD is used to identify regions of elevated intrinsic rotation at the initial time. Material coherence is then determined independently by testing finite-time contraction of candidate regions.

Workflow:

velocity field
    ->
trajectory integration
    ->
LAVD computation
    ->
boundary extraction
    ->
material advection
    ->
contraction test
    ->
LRCS identification

No new diagnostic is introduced. The method combines LAVD with a direct test of material contraction.

---

## Key idea

Advection does not define coherence.

Advection provides the material evolution of candidate boundaries. Coherence is established exclusively by verifying that the enclosed area decreases over the time interval.

A region is accepted as an LRCS if:

A(t1) / A(t0) < 1

---

## Files

lrcs.jl        Core implementation  
lrcs_run.jl    Driver script  
data/irma.nc   Example Hurricane Irma velocity data  
README.md      Documentation  

---

## Requirements

- Julia (>= 1.8 recommended)
- NetCDF support (for velocity input)
- Standard numerical libraries

---

## Quick start

Run:

julia lrcs_run.jl

This uses the included Hurricane Irma example:

data/irma.nc

---

## Usage

Typical workflow:

1. Define specifications in the driver:
   - DataSpec
   - DomainSpec
   - NumericsSpec
   - OutputSpec

2. Execute:
   include("lrcs_run.jl")

3. Outputs are written to the paths defined in OutputSpec.

---

## Input data

The code expects:

- velocity components on a grid
- time-resolved data
- consistent spatial coordinates

Typical units:

- velocity: m/s
- time: seconds
- coordinates: degrees (internally converted to meters)

---

## Output

The code produces:

- LAVD field at initial time
- selected boundary at t0
- advected boundary at t1
- optional intermediate boundary positions
- diagnostic quantities:
  - A(t0)
  - A(t1)
  - contraction ratio A1/A0

Outputs may include NetCDF files and figures depending on configuration.

---

## Internal workflow

### Candidate generation

LAVD is computed over the initial grid. Local maxima identify regions of elevated intrinsic rotation. Closed level sets are extracted around these maxima.

### Boundary selection

A boundary is selected among LAVD level sets based on:

- enclosure of a LAVD maximum
- minimum area threshold
- avoidance of excessive geometric irregularity

This step is geometric only.

### Material advection

Selected boundaries are advected over the time interval using a fourth-order Runge-Kutta scheme. Curves are periodically redistributed to maintain numerical stability.

### Contraction test

Let A(t) be the area enclosed by the boundary.

A candidate is accepted if:

A(t1) / A(t0) < 1

---

## Main components

Specifications:

- DataSpec: input file and variable names
- DomainSpec: spatial domain and time interval
- NumericsSpec: numerical parameters
- OutputSpec: output control

Core functions:

Velocity and trajectories:
- build_velocity_interpolant
- rk4_step
- advect_particles

LAVD:
- compute_lavd
- compute_vorticity

Boundary extraction:
- find_lavd_peaks
- extract_contours
- select_lrcs_curve

Material advection:
- advect_closed_curve
- redistribute_curve

Diagnostics:
- polygon_area

---

## Notes

- Initial localization (e.g., via vorticity or streamlines) is not required but may be used to focus analysis.
- The identification of LRCS is based on objective, material criteria.
- Not all LAVD maxima yield contracting regions.
- Instantaneous streamline patterns do not reliably indicate LRCS.

---

## License

MIT License (or specify your preferred license)

---

## Author

F. J. Beron-Vera  
University of Miami
