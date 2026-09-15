# diagnose_repulsion_3d
# nearest_neighbour_distances_3d
# repulsion_time_series_3d
# decision_breakdown_3d
# repulsion_turn_diagnostic_3d
# repulsion_norms_3d

# ============================================================
# Analysis (3D)
# ============================================================
function diagnose_repulsion_3d(
    pos::Vector{SVector3},
    vel::Vector{SVector3},
    P::CouzinParams;
    agent_ids::AbstractVector{Int} = [1, cld(length(pos), 2), length(pos)],
    max_list::Int = 10,
)
    rows = NamedTuple[]

    for i in agent_ids
        pi   = pos[i]
        vi_u = unit3(vel[i])

        rep = SVector3(0.0, 0.0, 0.0)
        ori = SVector3(0.0, 0.0, 0.0)
        att = SVector3(0.0, 0.0, 0.0)

        n_rep = 0
        n_ori = 0
        n_att = 0

        rep_dists = Float64[]
        all_dists = Float64[]

        @inbounds for j in eachindex(pos)
            j == i && continue

            d = displacement(pi, pos[j], P.L)
            r = norm(d)
            r == 0 && continue

            push!(all_dists, r)

            if r <= P.Zr
                rep -= d / r
                n_rep += 1
                push!(rep_dists, r)
                continue
            end

            d_u = d / r
            is_visible_3d(vi_u, d_u, P.blind_half_angle) || continue

            if r <= P.Zo
                ori += unit3(vel[j])
                n_ori += 1
            elseif r <= P.Za
                att += d_u
                n_att += 1
            end
        end

        nearest = isempty(all_dists) ? NaN : minimum(all_dists)

        push!(rows, (
            agent_id = i,
            nearest_neighbour = nearest,
            n_rep = n_rep,
            n_ori = n_ori,
            n_att = n_att,
            rep_norm = norm(rep),
            ori_norm = norm(ori),
            att_norm = norm(att),
            rep_dists = sort(rep_dists)[1:min(end, max_list)],
        ))
    end

    return rows
end

function nearest_neighbour_distances_3d(
    pos::Vector{SVector3},
    L::Float64;
    displacement,
)
    N = length(pos)
    nn = fill(Inf, N)

    @inbounds for i in 1:N
        pi = pos[i]
        for j in 1:N
            j == i && continue
            r = norm(displacement(pi, pos[j], L))
            if r < nn[i]
                nn[i] = r
            end
        end
    end

    return nn
end

function repulsion_time_series_3d(out, P::CouzinParams)
    T = length(out.pos_hist)

    mean_nrep = zeros(Float64, T)
    frac_with_rep = zeros(Float64, T)
    min_nn = zeros(Float64, T)
    median_nn = zeros(Float64, T)

    for k in 1:T
        pos = out.pos_hist[k]
        N = length(pos)

        nrep_vec = zeros(Int, N)
        nn = fill(Inf, N)

        @inbounds for i in 1:N
            pi = pos[i]
            for j in 1:N
                j == i && continue
                d = displacement(pi, pos[j], P.L)
                r = norm(d)
                r == 0 && continue

                if r < nn[i]
                    nn[i] = r
                end

                if r <= P.Zr
                    nrep_vec[i] += 1
                end
            end
        end

        mean_nrep[k] = mean(nrep_vec)
        frac_with_rep[k] = mean(nrep_vec .> 0)
        min_nn[k] = minimum(nn)
        median_nn[k] = median(nn)
    end

    return (
        t = out.t,
        mean_nrep = mean_nrep,
        frac_with_rep = frac_with_rep,
        min_nn = min_nn,
        median_nn = median_nn,
    )
end

function decision_breakdown_3d(i::Int,
                               pos::Vector{SVector3},
                               vel::Vector{SVector3},
                               P::CouzinParams)
    pi   = pos[i]
    vi_u = unit3(vel[i])

    rep = SVector3(0.0, 0.0, 0.0)
    ori = SVector3(0.0, 0.0, 0.0)
    att = SVector3(0.0, 0.0, 0.0)

    n_rep = 0
    n_ori = 0
    n_att = 0

    @inbounds for j in eachindex(pos)
        j == i && continue
        d = displacement(pi, pos[j], P.L)
        r = norm(d)
        r == 0 && continue

        if r <= P.Zr
            rep -= d / r
            n_rep += 1
            continue
        end

        d_u = d / r
        is_visible_3d(vi_u, d_u, P.blind_half_angle) || continue

        if r <= P.Zo
            ori += unit3(vel[j])
            n_ori += 1
        elseif r <= P.Za
            att += d_u
            n_att += 1
        end
    end

    desired =
        if n_rep > 0
            unit3(rep)
        else
            dir = SVector3(0.0, 0.0, 0.0)
            n_ori > 0 && (dir += ori)
            n_att > 0 && (dir += att)
            norm(dir) < 1e-12 ? vi_u : unit3(dir)
        end

    return (
        n_rep = n_rep,
        n_ori = n_ori,
        n_att = n_att,
        rep = rep,
        ori = ori,
        att = att,
        desired = desired,
        rep_dominates = n_rep > 0,
    )
end

function repulsion_turn_diagnostic_3d(i::Int,
                                      pos::Vector{SVector3},
                                      vel::Vector{SVector3},
                                      P::CouzinParams,
                                      dt::Float64)
    pi   = pos[i]
    vi_u = unit3(vel[i])

    rep = SVector3(0.0, 0.0, 0.0)
    n_rep = 0

    @inbounds for j in eachindex(pos)
        j == i && continue
        d = displacement(pi, pos[j], P.L)
        r = norm(d)
        r == 0 && continue

        if r <= P.Zr
            rep -= d / r
            n_rep += 1
        end
    end

    if n_rep == 0
        return (agent_id=i, n_rep=0)
    end

    desired = unit3(rep)
    c = clamp(dot(vi_u, desired), -1.0, 1.0)
    θ = acos(c)
    maxΔ = P.max_turn_rate * dt

    return (
        agent_id = i,
        n_rep = n_rep,
        theta_to_repulsion = θ,
        max_turn = maxΔ,
        turn_ratio = θ / maxΔ,
    )
end

function repulsion_norms_3d(pos::Vector{SVector3}, P::CouzinParams)
    vals = Float64[]
    for i in eachindex(pos)
        pi = pos[i]
        rep = SVector3(0.0, 0.0, 0.0)
        n_rep = 0

        @inbounds for j in eachindex(pos)
            j == i && continue
            d = displacement(pi, pos[j], P.L)
            r = norm(d)
            r == 0 && continue

            if r <= P.Zr
                rep -= d / r
                n_rep += 1
            end
        end

        push!(vals, n_rep > 0 ? norm(rep) : 0.0)
    end
    return vals
end
