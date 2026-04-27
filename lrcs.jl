module LRCS

# ================================================================
# lrcs.jl
# ================================================================
# Detection of Lagrangian Rotating Contracting Structures (LRCS)
# from two-dimensional velocity fields.
#
# ----------------------------------------------------------------
# What this code does
# ----------------------------------------------------------------
# - Computes the Lagrangian-averaged vorticity deviation (LAVD)
#   over a finite time interval [t0,t0+T].
# - Identifies candidate regions of elevated accumulated intrinsic
#   rotation.
# - Extracts closed material boundaries and keeps those that undergo
#   finite-time area contraction under advection.
# - Advects each extracted boundary from t0 to t0+T and to any user
#   requested selected times.
# - Writes a NetCDF output file and two diagnostic PNG figures.
#
# No new diagnostics are introduced: LAVD and direct material
# contraction are combined to isolate rotating contracting regions.
#
# ----------------------------------------------------------------
# Input data conventions (IMPORTANT)
# ----------------------------------------------------------------
# - Coordinates:
#       longitude, latitude in DEGREES
# - Velocity:
#       u, v in m/s (zonal, meridional components)
# - Time:
#       in SECONDS, with arbitrary origin
#       (only time differences are used)
#
# The code internally converts lon/lat to meters for interpolation,
# advection, area computation, and contraction testing.  The saved
# boundary output is in longitude/latitude degrees.
#
# Velocity arrays are expected as:
#       u(y, x, t), v(y, x, t)
# with:
#       y = latitude index
#       x = longitude index
#
# ----------------------------------------------------------------
# NetCDF output
# ----------------------------------------------------------------
# The file given by OutputSpec.outfile contains:
#
#   lon0(lat/lon grid coordinate)         degrees_east
#   lat0(lat/lon grid coordinate)         degrees_north
#   LAVD(lat0,lon0)                       accumulated vorticity deviation
#
#   lon_boundary(point,boundary)          extracted LRCS boundary at t0
#   lat_boundary(point,boundary)          extracted LRCS boundary at t0
#
#   advect_time(time)                     seconds after t0
#   lon_adv_boundary(point,boundary,time) boundary advected to t0+advect_time
#   lat_adv_boundary(point,boundary,time) boundary advected to t0+advect_time
#
# The boundary arrays are NaN-padded because different boundaries may have
# different numbers of points.  Boundary coordinates are saved in degrees,
# not meters.  Cartesian coordinates are used only internally.
#
# ----------------------------------------------------------------
# Figure output
# ----------------------------------------------------------------
# OutputSpec.plotfile is used only as a filename template.  If, for example,
# plotfile = "figure.png", the diagnostic images written are
#
#       figure_t0.png
#       figure_t0plusT.png
#
# The first image shows the extracted boundary at t0.  The second image shows
# the Dritschel-style redistributed material boundary advected from t0 to
# t0+T.  Both are plotted in longitude/latitude degrees.  No separate
# figure.png file is written.
#
# ----------------------------------------------------------------
# Driver pattern
# ----------------------------------------------------------------
# Create a separate script that includes this file, constructs DataSpec,
# DomainSpec, NumericsSpec, and OutputSpec, and then calls run_lrcs(cfg).
#
# ================================================================

using Dates
using Statistics
using LinearAlgebra
using Printf
using NCDatasets
using Contour
using CairoMakie
import Contour

export DataSpec, DomainSpec, NumericsSpec, OutputSpec, LRCSConfig, run_lrcs

# ============================================================
# Configuration containers
# ============================================================

"""
Stores NetCDF file and variable names.

Velocity components are assumed to be in m/s, longitude and latitude in degrees, and time in seconds unless decoded by NCDatasets.
"""
Base.@kwdef struct DataSpec
    infile::String
    lon_name::String
    lat_name::String
    time_name::String
    u_name::String
    v_name::String
end


"""
Defines the initial seed domain and the finite-time interval.

The longitude and latitude boxes may be given in either order; they are sorted internally before use.
"""
Base.@kwdef struct DomainSpec
    t0::DateTime
    T_seconds::Float64
    lon_box::Tuple{Float64,Float64}
    lat_box::Tuple{Float64,Float64}
end


"""
Controls LAVD resolution, time stepping, contour extraction, smoothing, contraction testing, and boundary redistribution.

`plot_quiver_step` is retained only for compatibility with older drivers; quiver arrows are not plotted or used.
"""
Base.@kwdef struct NumericsSpec
    lavd_N::Int
    dt_seconds::Float64
    min_region_area_m2::Float64
    nbuffer::Int
    dlevel::Float64
    peak_threshold::Float64
    peak_edge::Int
    peak_min_dist::Int
    smooth_passes::Int
    contraction_npts::Int
    plot_quiver_step::Int = 0
end


"""
Stores the NetCDF output path, figure filename template, and optional selected elapsed times after `t0` for advected-boundary output.
"""
Base.@kwdef struct OutputSpec
    outfile::String
    plotfile::String
    selected_time_offsets_seconds::Vector{Float64} = Float64[]
end


"""
Groups the data, domain, numerical, and output specifications passed to `run_lrcs`.
"""
Base.@kwdef struct LRCSConfig
    data::DataSpec
    domain::DomainSpec
    numerics::NumericsSpec
    output::OutputSpec
end


# ============================================================
# Internal data containers
# ============================================================

"""
Stores the velocity grid after NetCDF reading and coordinate conversion.

The arrays `u` and `v` use the internal layout `(ny,nx,nt)`.
"""
struct VelocityData
    lon::Vector{Float64}
    lat::Vector{Float64}
    x::Vector{Float64}
    y::Vector{Float64}
    t::Vector{Float64}
    datetime::Vector{DateTime}
    u::Array{Float64,3}
    v::Array{Float64,3}
    lon_ref::Float64
    lat_ref::Float64
end


# ============================================================
# Main driver
# ============================================================

"""
Run the full LRCS pipeline.

The function reads the velocity data, computes vorticity and LAVD, extracts contracting boundaries, advects them to selected output times, writes the NetCDF file, and saves diagnostic plots.
"""
function run_lrcs(cfg)
    vel = read_velocity(cfg.data)
    w = relative_vorticity(vel.x, vel.y, vel.t, vel.u, vel.v)

    x0, y0, lon0, lat0 = seed_grid(vel, cfg.domain, cfg.numerics.lavd_N)
    t0sec = datetime_to_internal_seconds(cfg.domain.t0)
    t1sec = t0sec + cfg.domain.T_seconds
    ttraj = collect(t0sec:cfg.numerics.dt_seconds:t1sec)
    if ttraj[end] < t1sec
        push!(ttraj, t1sec)
    end

    @info "Advecting seed grid and computing LAVD" nseeds=length(x0)*length(y0) nt=length(ttraj)
    LAVD = compute_lavd_direct(vel, w, x0, y0, ttraj)

    @info "Extracting contracting boundaries"
    boundaries = extract_lrcs_boundaries(LAVD, x0, y0, vel, ttraj, cfg.numerics)
    lonb, latb = boundaries_lonlat(boundaries, vel.lon_ref, vel.lat_ref)
    advect_offsets = boundary_output_times(cfg)
    lonadv, latadv = advect_boundaries_lonlat(boundaries, vel, t0sec, advect_offsets, cfg.numerics)

    @info "Writing NetCDF" outfile=cfg.output.outfile
    write_output(cfg.output.outfile, lon0, lat0, LAVD, lonb, latb, advect_offsets, lonadv, latadv, cfg, vel)

    @info "Writing boundary plots" plotfile_template=cfg.output.plotfile ntimes=length(advect_offsets)
    plot_advected_boundaries(cfg.output.plotfile, lon0, lat0, LAVD, advect_offsets, lonadv, latadv)

    return (LAVD=LAVD, lon0=lon0, lat0=lat0, boundaries=boundaries,
            lon_boundaries=lonb, lat_boundaries=latb,
            advect_time_offsets_seconds=advect_offsets,
            lon_adv_boundaries=lonadv, lat_adv_boundaries=latadv,
            output=cfg.output.outfile)
end


# ============================================================
# Reading and coordinates
# ============================================================

"""
Read longitude, latitude, time, and velocity components from NetCDF.

Latitude is flipped if necessary, coordinates are converted to a local Cartesian grid in meters, and velocity arrays are stored as `(ny,nx,nt)`.
"""
function read_velocity(spec)
    ds = NCDataset(spec.infile, "r")

    lon = Float64.(ds[spec.lon_name][:])
    lat = Float64.(ds[spec.lat_name][:])
    t = read_time_seconds(ds, spec.time_name)
    dtv = unix2datetime.(t)

    # Preserve NetCDF dimensionality. Do not use [:], which flattens.
    uin = Float64.(ds[spec.u_name][:, :, :])
    vin = Float64.(ds[spec.v_name][:, :, :])

    close(ds)

    if lat[1] > lat[end]
        lat = reverse(lat)
        uin = reverse_along_lat(uin, length(lon), length(lat), length(t))
        vin = reverse_along_lat(vin, length(lon), length(lat), length(t))
    end

    u = to_yxt(uin, length(lon), length(lat), length(t))
    v = to_yxt(vin, length(lon), length(lat), length(t))

    lon_ref = mean(lon)
    lat_ref = mean(lat)
    x = sph2x.(lon, lon_ref, lat_ref)
    y = sph2y.(lat, lat_ref)

    return VelocityData(lon, lat, x, y, t, dtv, u, v, lon_ref, lat_ref)
end


"""
Return time values in seconds.

DateTime input is converted to seconds since the Unix epoch. Numeric time is assumed to already be in seconds.
"""
function read_time_seconds(ds, name::String)
    v = ds[name]
    raw = v[:]
    if eltype(raw) <: DateTime
        return datetime_to_internal_seconds.(raw)
    end
    return Float64.(raw)
end


"""
Flip the latitude dimension of a velocity array before conversion to the internal `(ny,nx,nt)` layout.
"""
function reverse_along_lat(A, nx, ny, nt)
    if ndims(A) == 3 && size(A,1) == nx && size(A,2) == ny
        return A[:, end:-1:1, :]
    elseif ndims(A) == 3 && size(A,1) == ny && size(A,2) == nx
        return A[end:-1:1, :, :]
    else
        error("Cannot identify latitude dimension in velocity array of size $(size(A)).")
    end
end


"""
Convert common NetCDF velocity layouts to the internal `(ny,nx,nt)` layout.
"""
function to_yxt(A, nx, ny, nt)
    if ndims(A) != 3
        error("Velocity array must be 3-D; got size $(size(A)).")
    end
    if size(A) == (nx, ny, nt)
        return permutedims(A, (2,1,3))
    elseif size(A) == (ny, nx, nt)
        return A
    else
        error("Expected velocity dimensions (lon,lat,time)=($nx,$ny,$nt) or (lat,lon,time)=($ny,$nx,$nt); got $(size(A)).")
    end
end


const EARTH_R = 6371e3
const DEG2RAD = pi/180

"""
Convert longitude in degrees to local Cartesian `x` in meters.
"""
sph2x(lon, lon0, lat0) = EARTH_R * (lon - lon0) * DEG2RAD * cos(lat0 * DEG2RAD)

"""
Convert latitude in degrees to local Cartesian `y` in meters.
"""
sph2y(lat, lat0) = EARTH_R * (lat - lat0) * DEG2RAD

"""
Convert local Cartesian `x` in meters to longitude in degrees.
"""
xy2lon(x, lon0, lat0) = lon0 + x/(EARTH_R*cos(lat0*DEG2RAD))/DEG2RAD

"""
Convert local Cartesian `y` in meters to latitude in degrees.
"""
xy2lat(y, lat0) = lat0 + y/EARTH_R/DEG2RAD

"""
Convert a `DateTime` value to seconds since the Unix epoch.
"""
function datetime_to_internal_seconds(dt::DateTime)
    return Float64(Dates.value(dt - DateTime(1970,1,1,0,0,0))) / 1000.0
end


# ============================================================
# Numerics
# ============================================================

"""
Create the initial LAVD seed grid on the requested lon/lat box.

The number of grid points is adjusted to preserve the approximate Cartesian aspect ratio.
"""
function seed_grid(vel::VelocityData, dom, N::Int)
    lonmin, lonmax = extrema(dom.lon_box)
    latmin, latmax = extrema(dom.lat_box)
    xlo = sph2x(lonmin, vel.lon_ref, vel.lat_ref)
    xhi = sph2x(lonmax, vel.lon_ref, vel.lat_ref)
    ylo = sph2y(latmin, vel.lat_ref)
    yhi = sph2y(latmax, vel.lat_ref)

    fac = (xhi - xlo) / abs(yhi - ylo)
    if fac > 1
        nx0 = N
        ny0 = max(2, floor(Int, N/fac))
    else
        ny0 = N
        nx0 = max(2, floor(Int, N*fac))
    end

    x0 = collect(range(xlo, xhi, length=nx0))
    y0 = collect(range(ylo, yhi, length=ny0))
    lon0 = xy2lon.(x0, vel.lon_ref, vel.lat_ref)
    lat0 = xy2lat.(y0, vel.lat_ref)
    return x0, y0, lon0, lat0
end


"""
Return MATLAB-style meshgrid arrays with size `(ny,nx)`.
"""
function meshgrid_vec(x, y)
    X = repeat(reshape(x, 1, :), length(y), 1)
    Y = repeat(reshape(y, :, 1), 1, length(x))
    return X, Y
end


"""
Compute planar relative vorticity `omega = dv/dx - du/dy` on the local Cartesian grid.
"""
function relative_vorticity(x, y, t, u, v)
    ny, nx, nt = size(u)
    w = similar(u)
    invdx = derivative_inverse_widths(x)
    invdy = derivative_inverse_widths(y)
    @inbounds for k in 1:nt
        Threads.@threads for j in 1:ny
            jm = j == 1  ? 1 : j - 1
            jp = j == ny ? ny : j + 1
            sy = invdy[j]
            for i in 1:nx
                im = i == 1  ? 1 : i - 1
                ip = i == nx ? nx : i + 1
                dvdx = (v[j,ip,k] - v[j,im,k]) * invdx[i]
                dudy = (u[jp,i,k] - u[jm,i,k]) * sy
                w[j,i,k] = dvdx - dudy
            end
        end
    end
    return w
end


"""
Precompute reciprocal finite-difference widths for repeated first-derivative evaluations.
"""
function derivative_inverse_widths(x)
    n = length(x)
    invw = Vector{Float64}(undef, n)
    @inbounds begin
        invw[1] = 1 / (x[2] - x[1])
        for i in 2:n-1
            invw[i] = 1 / (x[i+1] - x[i-1])
        end
        invw[n] = 1 / (x[n] - x[n-1])
    end
    return invw
end


"""
Compute a first derivative along `x` with centered interior stencils and one-sided endpoint stencils.
"""
function ddx(A, x, j, i, k)
    nx = length(x)
    if i == 1
        return (A[j,i+1,k] - A[j,i,k])/(x[i+1]-x[i])
    elseif i == nx
        return (A[j,i,k] - A[j,i-1,k])/(x[i]-x[i-1])
    else
        return (A[j,i+1,k] - A[j,i-1,k])/(x[i+1]-x[i-1])
    end
end


"""
Compute a first derivative along `y` with centered interior stencils and one-sided endpoint stencils.
"""
function ddy(A, y, j, i, k)
    ny = length(y)
    if j == 1
        return (A[j+1,i,k] - A[j,i,k])/(y[j+1]-y[j])
    elseif j == ny
        return (A[j,i,k] - A[j-1,i,k])/(y[j]-y[j-1])
    else
        return (A[j+1,i,k] - A[j-1,i,k])/(y[j+1]-y[j-1])
    end
end


"""
Evaluate a trilinear interpolant on regular `x`, `y`, and `t` grids.

The lookup assumes monotone, nearly uniform grids and returns `NaN` outside the grid.
"""
@inline function interp3_regular(xgrid, ygrid, tgrid, A, x, y, tt)
    ix = bracket_index_regular(xgrid, x)
    iy = bracket_index_regular(ygrid, y)
    it = bracket_index_regular(tgrid, tt)
    if ix == 0 || iy == 0 || it == 0
        return NaN
    end

    @inbounds begin
        x1 = xgrid[ix];     x2 = xgrid[ix+1]
        y1 = ygrid[iy];     y2 = ygrid[iy+1]
        t1 = tgrid[it];     t2 = tgrid[it+1]
        ax = (x - x1)/(x2 - x1)
        ay = (y - y1)/(y2 - y1)
        at = (tt - t1)/(t2 - t1)

        c000 = A[iy,ix,it];     c100 = A[iy,ix+1,it]
        c010 = A[iy+1,ix,it];   c110 = A[iy+1,ix+1,it]
        c001 = A[iy,ix,it+1];   c101 = A[iy,ix+1,it+1]
        c011 = A[iy+1,ix,it+1]; c111 = A[iy+1,ix+1,it+1]

        c00 = (1-ax)*c000 + ax*c100
        c10 = (1-ax)*c010 + ax*c110
        c01 = (1-ax)*c001 + ax*c101
        c11 = (1-ax)*c011 + ax*c111
        c0 = (1-ay)*c00 + ay*c10
        c1 = (1-ay)*c01 + ay*c11
        return (1-at)*c0 + at*c1
    end
end


"""
Return the lower interpolation-cell index for a monotone regular grid.

The function returns zero when the query value lies outside the grid.
"""
@inline function bracket_index_regular(grid, val)
    n = length(grid)
    g1 = grid[1]
    gn = grid[end]
    if !(g1 <= val <= gn)
        return 0
    end
    h = (gn - g1)/(n - 1)
    i = floor(Int, (val - g1)/h) + 1
    if i < 1
        return 0
    elseif i >= n
        return n - 1
    else
        return i
    end
end


"""
Evaluate the Cartesian velocity field at one point and time.
"""
function velocity_at(vel::VelocityData, x, y, tt)
    return (interp3_regular(vel.x, vel.y, vel.t, vel.u, x, y, tt),
            interp3_regular(vel.x, vel.y, vel.t, vel.v, x, y, tt))
end


"""
Advance one particle by one fixed Runge-Kutta step.
"""
function rk4_step(vel, x, y, tt, h)
    u1,v1 = velocity_at(vel, x, y, tt)
    if !isfinite(u1+v1); return NaN, NaN; end
    u2,v2 = velocity_at(vel, x + 0.5h*u1, y + 0.5h*v1, tt + 0.5h)
    if !isfinite(u2+v2); return NaN, NaN; end
    u3,v3 = velocity_at(vel, x + 0.5h*u2, y + 0.5h*v2, tt + 0.5h)
    if !isfinite(u3+v3); return NaN, NaN; end
    u4,v4 = velocity_at(vel, x + h*u3, y + h*v3, tt + h)
    if !isfinite(u4+v4); return NaN, NaN; end
    return x + h*(u1 + 2u2 + 2u3 + u4)/6, y + h*(v1 + 2v2 + 2v3 + v4)/6
end


"""
Advance many particles over a prescribed time vector using fixed-step RK4.
"""
function advect_points(vel::VelocityData, x0::Vector, y0::Vector, tspan::Vector)
    n = length(x0)
    nt = length(tspan)
    xt = fill(NaN, n, nt)
    yt = fill(NaN, n, nt)
    xt[:,1] .= x0
    yt[:,1] .= y0
    for k in 1:nt-1
        h = tspan[k+1] - tspan[k]
        Threads.@threads for p in 1:n
            if isfinite(xt[p,k] + yt[p,k])
                xt[p,k+1], yt[p,k+1] = rk4_step(vel, xt[p,k], yt[p,k], tspan[k], h)
            end
        end
    end
    return xt, yt
end



"""
Compute LAVD by advecting the seed grid and accumulating the vorticity-deviation integral in one pass.

This avoids storing the full trajectory array.
"""
function compute_lavd_direct(vel, w, x0, y0, ttraj)
    X0, Y0 = meshgrid_vec(x0, y0)
    xcur = collect(vec(X0))
    ycur = collect(vec(Y0))
    xnew = similar(xcur)
    ynew = similar(ycur)

    n = length(xcur)
    nt = length(ttraj)
    prev = fill(NaN, n)
    curr = fill(NaN, n)
    acc = zeros(Float64, n)
    alive = trues(n)

    for k in 1:nt
        tt = ttraj[k]
        Threads.@threads for p in 1:n
            @inbounds curr[p] = alive[p] ? interp3_regular(vel.x, vel.y, vel.t, w, xcur[p], ycur[p], tt) : NaN
        end

        m = finite_mean(curr)
        Threads.@threads for p in 1:n
            @inbounds curr[p] = isfinite(curr[p]) && isfinite(m) ? abs(curr[p] - m) : NaN
        end

        if k > 1
            hprev = ttraj[k] - ttraj[k-1]
            Threads.@threads for p in 1:n
                @inbounds begin
                    if alive[p] && isfinite(prev[p]) && isfinite(curr[p])
                        acc[p] += 0.5 * (prev[p] + curr[p]) * hprev
                    else
                        alive[p] = false
                    end
                end
            end
        end

        if k < nt
            h = ttraj[k+1] - ttraj[k]
            Threads.@threads for p in 1:n
                @inbounds begin
                    if alive[p] && isfinite(xcur[p] + ycur[p])
                        xnew[p], ynew[p] = rk4_step(vel, xcur[p], ycur[p], tt, h)
                    else
                        xnew[p] = NaN
                        ynew[p] = NaN
                    end
                end
            end
            xcur, xnew = xnew, xcur
            ycur, ynew = ynew, ycur
        end

        prev, curr = curr, prev
    end

    lavd = Vector{Float64}(undef, n)
    Threads.@threads for p in 1:n
        @inbounds lavd[p] = alive[p] ? abs(acc[p]) : NaN
    end
    return reshape(lavd, length(y0), length(x0))
end


"""
Evaluate `|omega - mean(omega)|` along stored trajectories and integrate it over time with the trapezoidal rule.
"""
function compute_lavd(vel, w, xtraj, ytraj, ttraj, x0, y0)
    n, nt = size(xtraj)
    prev = fill(NaN, n)
    curr = fill(NaN, n)
    acc = zeros(Float64, n)
    alive = trues(n)

    for k in 1:nt
        Threads.@threads for p in 1:n
            curr[p] = interp3_regular(vel.x, vel.y, vel.t, w, xtraj[p,k], ytraj[p,k], ttraj[k])
        end
        m = finite_mean(curr)
        Threads.@threads for p in 1:n
            @inbounds curr[p] = isfinite(curr[p]) && isfinite(m) ? abs(curr[p] - m) : NaN
        end

        if k > 1
            h = ttraj[k] - ttraj[k-1]
            Threads.@threads for p in 1:n
                @inbounds begin
                    if alive[p] && isfinite(prev[p]) && isfinite(curr[p])
                        acc[p] += 0.5 * (prev[p] + curr[p]) * h
                    else
                        alive[p] = false
                    end
                end
            end
        end
        prev, curr = curr, prev
    end

    lavd = Vector{Float64}(undef, n)
    Threads.@threads for p in 1:n
        @inbounds lavd[p] = alive[p] ? abs(acc[p]) : NaN
    end
    return reshape(lavd, length(y0), length(x0))
end


"""
Return an iterator over finite entries of a vector.
"""
skipmissing_nan(v) = (x for x in v if isfinite(x))

"""
Return the mean of finite entries without allocating an intermediate array.
"""
function finite_mean(v)
    s = 0.0
    n = 0
    @inbounds for x in v
        if isfinite(x)
            s += x
            n += 1
        end
    end
    return n == 0 ? NaN : s / n
end


# ============================================================
# Boundary extraction
# ============================================================

"""
Find LAVD peaks, select contracting closed contours around them, and remove duplicate boundaries by peak containment.
"""
function extract_lrcs_boundaries(LAVD, x0, y0, vel, ttraj, num)
    f = smooth_field(LAVD, num.smooth_passes)
    f ./= maximum(skipmissing_nan(vec(f)))
    peakmask = findpeaks2d(f, num.peak_threshold, num.peak_edge, num.peak_min_dist)
    X, Y = meshgrid_vec(x0, y0)
    peaks = findall(peakmask)
    candidates = Vector{Matrix{Float64}}()
    areas = Float64[]
    peakxy = Tuple{Float64,Float64}[]

    for (ip, I) in enumerate(peaks)
        j, i = Tuple(I)
        @info "LRCS peak" peak=ip total=length(peaks)
        curve = select_lrcs_curve(x0, y0, f, X[j,i], Y[j,i], vel, ttraj, num)
        if curve !== nothing
            push!(candidates, curve)
            push!(areas, polyarea(curve[1,:], curve[2,:]))
            push!(peakxy, (X[j,i], Y[j,i]))
        end
    end

    if isempty(candidates)
        @info "Found 0 non-duplicate boundaries"
        return Matrix{Float64}[]
    end

    order = sortperm(areas, rev=true)
    keep = Matrix{Float64}[]
    for idx in order
        px, py = peakxy[idx]
        isdup = any(point_in_poly(px, py, c[1,:], c[2,:]) for c in keep)
        if !isdup
            push!(keep, candidates[idx])
        end
    end
    @info "Found $(length(keep)) non-duplicate boundaries"
    return keep
end


"""
Apply repeated 3-by-3 finite-value averaging to a scalar field.
"""
function smooth_field(F, npass)
    G = copy(F)
    H = similar(G)
    ny, nx = size(G)
    for _ in 1:npass
        @inbounds for j in 1:ny, i in 1:nx
            s = 0.0
            n = 0
            for jj in max(1,j-1):min(ny,j+1), ii in max(1,i-1):min(nx,i+1)
                val = G[jj,ii]
                if isfinite(val)
                    s += val
                    n += 1
                end
            end
            H[j,i] = n == 0 ? NaN : s / n
        end
        G, H = H, G
    end
    return G
end


"""
Detect strict local maxima and apply greedy distance-based nonmaximum suppression.
"""
function findpeaks2d(f, threshold, edg, minDist)
    nr, nc = size(f)
    mask = falses(nr,nc)
    for j in 1+edg:nr-edg, i in 1+edg:nc-edg
        fij = f[j,i]
        if !isfinite(fij) || fij < threshold
            continue
        end
        good = true
        for jj in j-1:j+1, ii in i-1:i+1
            if (jj != j || ii != i) && fij <= f[jj,ii]
                good = false
                break
            end
        end
        mask[j,i] = good
    end
    inds = findall(mask)
    isempty(inds) && return mask
    vals = [f[I] for I in inds]
    order = sortperm(vals, rev=true)
    keep = falses(length(inds))
    taken = falses(length(inds))
    for q in order
        taken[q] && continue
        keep[q] = true
        jq, iq = Tuple(inds[q])
        for r in eachindex(inds)
            jr, ir = Tuple(inds[r])
            if hypot(jr-jq, ir-iq) < minDist
                taken[r] = true
            end
        end
    end
    out = falses(nr,nc)
    for (k,I) in enumerate(inds)
        keep[k] && (out[I] = true)
    end
    return out
end


"""
Select a contracting LRCS boundary around one LAVD peak.

The function scans normalized LAVD contours, scores closed curves by mean excess over the contour level, applies the inward buffer, and returns the first candidate satisfying finite-time area contraction.
"""
function select_lrcs_curve(x0, y0, f, xmax, ymax, vel, ttraj, num)
    levels = collect(0:num.dlevel:1)
    X, Y = meshgrid_vec(x0, y0)
    curves = Matrix{Float64}[]
    zlev = Float64[]
    areas = Float64[]
    scores = Float64[]

    for lev in levels
        cs = Contour.contours(x0, y0, f', [lev])
        for cl in Contour.levels(cs)
            for line in Contour.lines(cl)
                xs, ys = Contour.coordinates(line)
                xvec = collect(Float64.(xs)); yvec = collect(Float64.(ys))
                length(xvec) < 4 && continue
                hypot(xvec[end]-xvec[1], yvec[end]-yvec[1]) > 1e-8 && continue
                point_in_poly(xmax, ymax, xvec, yvec) || continue
                A = polyarea(xvec, yvec)
                isfinite(A) && A >= num.min_region_area_m2 || continue
                sscore = 0.0
                nscore = 0
                for idx in eachindex(X)
                    if point_in_poly(X[idx], Y[idx], xvec, yvec)
                        val = f[idx]
                        if isfinite(val)
                            sscore += val - lev
                            nscore += 1
                        end
                    end
                end
                nscore == 0 && continue
                score = sscore / nscore
                isfinite(score) || continue
                push!(curves, permutedims(hcat(xvec, yvec)))
                push!(zlev, lev)
                push!(areas, A)
                push!(scores, score)
            end
        end
    end

    isempty(curves) && return nothing
    Iarea = sortperm(areas, rev=true)
    imax_local = argmax(scores)
    ipos = findfirst(==(imax_local), Iarea)
    jstart = min(ipos + num.nbuffer, length(Iarea))

    for jpos in jstart:length(Iarea)
        ibest = Iarea[jpos]
        ratio = area_contraction_ratio(curves[ibest], vel, ttraj, num.contraction_npts)
        if isfinite(ratio) && ratio < 1.0 - 1e-8
            @info "Selected contracting contour" level=zlev[ibest] area0_m2=areas[ibest] contraction_ratio=ratio
            return regularize_closed_curve(curves[ibest], num.contraction_npts, 2; preserve_area=true)
        end
    end
    @info "No contracting contour found around peak" nbuffer=num.nbuffer
    return nothing
end


"""
Compute the unsigned area enclosed by an ordered planar polygon.
"""
function polyarea(x, y)
    n = length(x)
    n < 3 && return NaN
    s = 0.0
    for i in 1:n
        ip = i == n ? 1 : i+1
        s += x[i]*y[ip] - x[ip]*y[i]
    end
    return abs(s)/2
end


"""
Test whether a point lies inside a polygon using ray crossing.
"""
function point_in_poly(x, y, xv, yv)
    inside = false
    j = length(xv)
    for i in eachindex(xv)
        if ((yv[i] > y) != (yv[j] > y)) &&
           (x < (xv[j]-xv[i])*(y-yv[i])/(yv[j]-yv[i] + eps()) + xv[i])
            inside = !inside
        end
        j = i
    end
    return inside
end


"""
Return true when the advected material polygon encloses less area at the final time than at the initial time.
"""
function area_contracts(curve, vel, ttraj, npts)
    ratio = area_contraction_ratio(curve, vel, ttraj, npts)
    return isfinite(ratio) && ratio < 1.0 - 1e-8
end


"""
Compute `|U(t0+T)|/|U(t0)|` for a closed material curve.

The curve is redistributed between RK4 steps so that the final polygon remains ordered before the area is computed.
"""
function area_contraction_ratio(curve, vel, ttraj, npts)
    c = regularize_closed_curve(curve, npts, 2; preserve_area=true)
    A0 = polyarea(c[1,:], c[2,:])
    (!isfinite(A0) || A0 <= 0) && return NaN

    ca = advect_material_curve(c, vel, collect(ttraj), npts; smooth_passes=0)
    ca === nothing && return NaN

    Af = polyarea(ca[1,:], ca[2,:])
    return isfinite(Af) ? Af / A0 : NaN
end


"""
Advect a closed curve over a supplied time grid using contour-style redistribution after each RK4 step.
"""
function advect_closed_curve(xc, yc, vel, ttraj, npts)
    c = regularize_closed_curve(permutedims(hcat(collect(xc), collect(yc))), npts, 2; preserve_area=true)
    ca = advect_material_curve(c, vel, collect(ttraj), npts; smooth_passes=0)
    ca === nothing && return fill(NaN, npts), fill(NaN, npts)
    return collect(ca[1,:]), collect(ca[2,:])
end


"""
Advance a closed material curve with fixed-step RK4 and redistribute it uniformly in arclength after every step.

This MATLAB-style contour-advection step prevents point clustering, artificial chords, and loss of curve ordering in saved boundaries and contraction-area calculations.
"""
function advect_material_curve(curve, vel, tspan, npts; smooth_passes=0)
    c = resample_closed_curve(curve, npts)
    x = collect(c[1,:])
    y = collect(c[2,:])
    length(tspan) == 1 && return c

    for k in 1:length(tspan)-1
        h = tspan[k+1] - tspan[k]
        xnew = similar(x)
        ynew = similar(y)
        Threads.@threads for p in eachindex(x)
            if isfinite(x[p] + y[p])
                xnew[p], ynew[p] = rk4_step(vel, x[p], y[p], tspan[k], h)
            else
                xnew[p] = NaN
                ynew[p] = NaN
            end
        end
        if any(!isfinite, xnew) || any(!isfinite, ynew)
            return nothing
        end
        c = resample_closed_curve(permutedims(hcat(xnew, ynew)), npts)
        if smooth_passes > 0
            c = regularize_closed_curve(c, npts, smooth_passes; preserve_area=false)
        end
        x = collect(c[1,:])
        y = collect(c[2,:])
    end
    return permutedims(hcat(x, y))
end



"""
Redistribute a closed curve and apply periodic binomial smoothing.

When `preserve_area=true`, the smoothed curve is rescaled about its centroid to keep the original enclosed area.
"""
function regularize_closed_curve(curve, npts, npass; preserve_area=false)
    c = resample_closed_curve(curve, npts)
    npass <= 0 && return c

    A0 = polyarea(c[1,:], c[2,:])
    x = collect(c[1,:])
    y = collect(c[2,:])
    n = length(x)
    n < 4 && return c

    for _ in 1:npass
        xo = copy(x)
        yo = copy(y)
        @inbounds for i in 1:n
            im = i == 1 ? n : i - 1
            ip = i == n ? 1 : i + 1
            x[i] = 0.25 * xo[im] + 0.5 * xo[i] + 0.25 * xo[ip]
            y[i] = 0.25 * yo[im] + 0.5 * yo[i] + 0.25 * yo[ip]
        end
    end

    cnew = permutedims(hcat(x, y))
    if preserve_area && isfinite(A0) && A0 > 0
        A1 = polyarea(cnew[1,:], cnew[2,:])
        if isfinite(A1) && A1 > 0
            cx = mean(cnew[1,:])
            cy = mean(cnew[2,:])
            fac = sqrt(A0 / A1)
            cnew[1,:] .= cx .+ fac .* (cnew[1,:] .- cx)
            cnew[2,:] .= cy .+ fac .* (cnew[2,:] .- cy)
        end
    end
    return cnew
end


"""
Redistribute a closed curve uniformly in arclength.
"""
function resample_closed_curve(curve, npts)
    x = collect(curve[1,:]); y = collect(curve[2,:])
    if hypot(x[end]-x[1], y[end]-y[1]) < 1e-12
        pop!(x); pop!(y)
    end
    length(x) < 3 && return curve
    xc = [x; x[1]]; yc = [y; y[1]]
    s = zeros(length(xc))
    for i in 2:length(xc)
        s[i] = s[i-1] + hypot(xc[i]-xc[i-1], yc[i]-yc[i-1])
    end
    L = s[end]
    L <= 0 && return curve
    sq = range(0, L, length=npts+1)[1:end-1]
    xr = interp1_linear(s, xc, collect(sq))
    yr = interp1_linear(s, yc, collect(sq))
    return permutedims(hcat(xr, yr))
end


"""
Perform one-dimensional linear interpolation on a monotone grid.

Values outside the interpolation interval are clamped to the nearest endpoint.
"""
function interp1_linear(x, y, xq)
    out = similar(xq, Float64)
    n = length(x)
    @inbounds for k in eachindex(xq)
        q = xq[k]
        if q <= x[1]
            out[k] = y[1]
        elseif q >= x[n]
            out[k] = y[n]
        else
            i = searchsortedlast(x, q)
            i = min(max(i, 1), n - 1)
            a = (q - x[i])/(x[i+1] - x[i])
            out[k] = (1-a)*y[i] + a*y[i+1]
        end
    end
    return out
end


"""
Convert extracted Cartesian boundaries from meters to longitude/latitude degrees.
"""
function boundaries_lonlat(boundaries, lon_ref, lat_ref)
    lonb = Vector{Vector{Float64}}()
    latb = Vector{Vector{Float64}}()
    for c in boundaries
        push!(lonb, xy2lon.(c[1,:], lon_ref, lat_ref))
        push!(latb, xy2lat.(c[2,:], lat_ref))
    end
    return lonb, latb
end


# ============================================================
# Output and plotting
# ============================================================

"""
Return elapsed output times in seconds after `t0`.

The result always includes zero and `T`, plus any selected times requested in `OutputSpec`.
"""
function boundary_output_times(cfg)
    vals = Float64[0.0, cfg.domain.T_seconds]
    append!(vals, cfg.output.selected_time_offsets_seconds)
    vals = [x for x in vals if isfinite(x) && 0.0 <= x <= cfg.domain.T_seconds]
    sort!(unique!(vals))
    return vals
end


"""
Advect each extracted boundary to the requested output times and convert the result to longitude/latitude degrees for storage and plotting.
"""
function advect_boundaries_lonlat(boundaries, vel, t0sec, offsets, num)
    nb = length(boundaries)
    nt = length(offsets)
    npts = output_boundary_npts(num)
    Lon = fill(NaN, npts, max(nb,1), nt)
    Lat = fill(NaN, npts, max(nb,1), nt)

    for b in 1:nb
        c = regularize_closed_curve(boundaries[b], npts, 3; preserve_area=true)
        xinit = collect(c[1,:])
        yinit = collect(c[2,:])
        for k in 1:nt
            dtout = offsets[k]
            if dtout == 0.0
                xa = xinit
                ya = yinit
            else
                tspan = advection_tspan(t0sec, t0sec + dtout, num.dt_seconds)
                ca = advect_material_curve(c, vel, tspan, npts; smooth_passes=1)
                if ca === nothing
                    xa = fill(NaN, npts)
                    ya = fill(NaN, npts)
                else
                    ca = regularize_closed_curve(ca, npts, 1; preserve_area=false)
                    xa = collect(ca[1,:])
                    ya = collect(ca[2,:])
                end
            end
            Lon[:,b,k] .= xy2lon.(xa, vel.lon_ref, vel.lat_ref)
            Lat[:,b,k] .= xy2lat.(ya, vel.lat_ref)
        end
    end
    return Lon, Lat
end


"""
Return the number of points used for saved and plotted boundaries.

This output resolution does not change the contraction test.
"""
function output_boundary_npts(num)
    return max(8 * num.contraction_npts, 2400)
end


"""
Build a fixed-step time vector that lands exactly on the requested final time.
"""
function advection_tspan(t0, t1, dt)
    if t1 == t0
        return [t0]
    end
    h = abs(dt)
    ts = collect(t0:h:t1)
    if ts[end] < t1
        push!(ts, t1)
    end
    return ts
end


"""
Write the LAVD grid, extracted boundaries at `t0`, and advected boundaries to NetCDF.

Boundary coordinates are stored in longitude/latitude degrees and NaN-padded along the point dimension.
"""
function write_output(outfile, lon0, lat0, LAVD, lonb, latb, offsets, lonadv, latadv, cfg, vel)
    isfile(outfile) && rm(outfile)
    ds = NCDataset(outfile, "c")
    defDim(ds, "lon0", length(lon0))
    defDim(ds, "lat0", length(lat0))
    nb = length(lonb)
    nmax0 = nb == 0 ? 1 : maximum(length(c) for c in lonb)
    nmaxa = size(lonadv, 1)
    nmax = max(nmax0, nmaxa)
    defDim(ds, "boundary_point", nmax)
    defDim(ds, "boundary", max(nb,1))
    defDim(ds, "advect_time", length(offsets))

    ds.attrib["title"] = "LRCS LAVD, extracted boundaries, and advected boundaries"
    ds.attrib["input_file"] = cfg.data.infile
    ds.attrib["t0"] = string(cfg.domain.t0)
    ds.attrib["T_seconds"] = cfg.domain.T_seconds
    ds.attrib["lon_ref"] = vel.lon_ref
    ds.attrib["lat_ref"] = vel.lat_ref
    ds.attrib["summary"] = "Boundary coordinates are saved in longitude/latitude degrees. Cartesian meters are used only internally for advection and area contraction."

    vlon = defVar(ds, "lon0", Float64, ("lon0",))
    vlon[:] = lon0
    vlon.attrib["units"] = "degrees_east"
    vlon.attrib["long_name"] = "initial longitude grid for LAVD"

    vlat = defVar(ds, "lat0", Float64, ("lat0",))
    vlat[:] = lat0
    vlat.attrib["units"] = "degrees_north"
    vlat.attrib["long_name"] = "initial latitude grid for LAVD"

    vl = defVar(ds, "LAVD", Float64, ("lat0", "lon0"))
    vl[:,:] = LAVD
    vl.attrib["long_name"] = "Lagrangian-averaged vorticity deviation accumulated over [t0,t0+T]"
    vl.attrib["units"] = "dimensionless"

    vt = defVar(ds, "advect_time", Float64, ("advect_time",))
    vt[:] = offsets
    vt.attrib["units"] = "seconds since t0"
    vt.attrib["long_name"] = "elapsed output time for advected boundaries"

    Lonb = fill(NaN, nmax, max(nb,1))
    Latb = fill(NaN, nmax, max(nb,1))
    for k in 1:nb
        n = length(lonb[k])
        Lonb[1:n,k] .= lonb[k]
        Latb[1:n,k] .= latb[k]
    end

    vblon = defVar(ds, "lon_boundary", Float64, ("boundary_point", "boundary"))
    vblat = defVar(ds, "lat_boundary", Float64, ("boundary_point", "boundary"))
    vblon[:,:] = Lonb
    vblat[:,:] = Latb
    vblon.attrib["units"] = "degrees_east"
    vblat.attrib["units"] = "degrees_north"
    vblon.attrib["long_name"] = "extracted LRCS boundary longitude at t0"
    vblat.attrib["long_name"] = "extracted LRCS boundary latitude at t0"

    Lonadv = fill(NaN, nmax, max(nb,1), length(offsets))
    Latadv = fill(NaN, nmax, max(nb,1), length(offsets))
    if nb > 0
        Lonadv[1:size(lonadv,1), 1:nb, :] .= lonadv[:,1:nb,:]
        Latadv[1:size(latadv,1), 1:nb, :] .= latadv[:,1:nb,:]
    end

    valon = defVar(ds, "lon_adv_boundary", Float64, ("boundary_point", "boundary", "advect_time"))
    valat = defVar(ds, "lat_adv_boundary", Float64, ("boundary_point", "boundary", "advect_time"))
    valon[:,:,:] = Lonadv
    valat[:,:,:] = Latadv
    valon.attrib["units"] = "degrees_east"
    valat.attrib["units"] = "degrees_north"
    valon.attrib["long_name"] = "LRCS boundary longitude advected from t0 to t0+advect_time"
    valat.attrib["long_name"] = "LRCS boundary latitude advected from t0 to t0+advect_time"

    close(ds)
end


"""
Save diagnostic images of LAVD and extracted or advected boundaries in longitude/latitude degrees.
"""
function plot_advected_boundaries(plotfile, lon0, lat0, LAVD, offsets, lonadv, latadv)
    T = maximum(offsets)
    for k in eachindex(offsets)
        fig = Figure(size=(1000,800))
        title = offsets[k] == 0.0 ? "LRCS boundaries at t0" : @sprintf("LRCS boundaries advected to t0 + %.0f s", offsets[k])
        ax = Axis(fig[1,1], xlabel="longitude", ylabel="latitude", title=title)
        hm = heatmap!(ax, lon0, lat0, permutedims(LAVD), colormap=:viridis)
        Colorbar(fig[1,2], hm, label="LAVD at t0")
        nb = size(lonadv, 2)
        for b in 1:nb
            xa = lonadv[:,b,k]
            ya = latadv[:,b,k]
            inds = findall(i -> isfinite(xa[i] + ya[i]), eachindex(xa))
            isempty(inds) && continue
            xp, yp = close_plot_curve(xa[inds], ya[inds])
            CairoMakie.lines!(ax, xp, yp, color=:white, linewidth=3)
            CairoMakie.lines!(ax, xp, yp, color=:black, linewidth=1)
        end
        xlims!(ax, extrema(lon0)...)
        ylims!(ax, extrema(lat0)...)
        save(boundary_plotfile(plotfile, offsets[k], T), fig)
    end
end


"""
Append the first point to the end of a plotted boundary.
"""
function close_plot_curve(x, y)
    isempty(x) && return x, y
    return [collect(x); x[1]], [collect(y); y[1]]
end


"""
Construct diagnostic figure filenames for initial, final, and intermediate output times.
"""
function boundary_plotfile(plotfile, offset, T)
    dir = dirname(plotfile)
    base = basename(plotfile)
    stem, ext = splitext(base)
    ext = isempty(ext) ? ".png" : ext
    if isapprox(offset, 0.0; atol=1e-9)
        name = string(stem, "_t0", ext)
    elseif isapprox(offset, T; atol=1e-9)
        name = string(stem, "_t0plusT", ext)
    else
        name = @sprintf("%s_t%06.0fs%s", stem, offset, ext)
    end
    return isempty(dir) ? name : joinpath(dir, name)
end


end # module

