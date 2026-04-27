#!/usr/bin/env julia

include("lrcs.jl")

using Dates
using .LRCS

cfg = LRCS.LRCSConfig(
    data = LRCS.DataSpec(
        infile = "data/irma.nc",
        lon_name = "longitude",
        lat_name = "latitude",
        time_name = "time",
        u_name = "u",
        v_name = "v",
    ),
    domain = LRCS.DomainSpec(
        t0 = DateTime(2017, 9, 6, 3, 0, 0),
        T_seconds = 36.0 * 3600.0,
        lon_box = (-72.0, -58.0),
        lat_box = (14.0, 26.0),
    ),
    numerics = LRCS.NumericsSpec(
        lavd_N = 256,
        dt_seconds = 0.1 * 3600.0,
        min_region_area_m2 = pi * (25e3)^2,
        nbuffer = 0,
        dlevel = 0.1,
        peak_threshold = 0.1,
        peak_edge = 3,
        peak_min_dist = 15,
        smooth_passes = 1,
        contraction_npts = 300,
    ),
    output = LRCS.OutputSpec(
        outfile = "output.nc",
        plotfile = "figure.png",
    ),
)

result = LRCS.run_lrcs(cfg)

println("Wrote: ", result.output)
println("Number of boundaries: ", length(result.boundaries))
