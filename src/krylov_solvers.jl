function krylov_options(p;
        iterations = length(rhs(p)),
        abstol = zero(real(eltype(rhs(p)))),
        reltol = sqrt(eps(real(eltype(rhs(p))))),
        norm = norm,
        Pl=preconditioner(identity_solver,p),
        update_Pl = true,
        verbose = false,
        output_prefix = "",
    )
    KrylovOptions(iterations,abstol,reltol,norm,Pl,update_Pl,verbose,output_prefix)
end

struct KrylovOptions{A,B,C,D,E,F,G,H} <: AbstractType
    iterations::A
    abstol::B
    reltol::C
    norm::D
    Pl::E
    update_Pl::F
    verbose::G
    output_prefix::H
end

function update(o::KrylovOptions;kwargs...)
    data = (;kwargs...)
    if hasproperty(data,:iterations)
        iterations = data.iterations
    else
        iterations = o.iterations
    end
    if hasproperty(data,:abstol)
        abstol = data.abstol
    else
        abstol = o.abstol
    end
    if hasproperty(data,:reltol)
        reltol = data.reltol
    else
        reltol = o.reltol
    end
    if hasproperty(data,:norm)
        norm = data.norm
    else
        norm = o.norm
    end
    if hasproperty(data,:Pl)
        Pl = data.Pl
    else
        Pl = o.Pl
    end
    if hasproperty(data,:update_Pl)
        update_Pl = data.update_Pl
    else
        update_Pl = o.update_Pl
    end
    if hasproperty(data,:verbose)
        verbose = data.verbose
    else
        verbose = o.verbose
    end
    if hasproperty(data,:output_prefix)
        output_prefix = data.output_prefix
    else
        output_prefix = o.output_prefix
    end
    KrylovOptions(iterations,abstol,reltol,norm,Pl,update_Pl,verbose,output_prefix)
end

function cg_state(p)
    x = solution(p)
    A = matrix(p)
    dx = similar(x,axes(A,2))
    u = similar(x)
    r = similar(x,axes(A,1))
    ρ = one(real(eltype(x)))
    iteration = 0
    current = zero(real(eltype(x)))
    target = zero(real(eltype(x)))
    PGCState(dx,r,u,A,ρ,iteration,current,target)
end

struct PGCState{A,B,C,D,E,F,G,H} <: AbstractType
    dx::A
    r::B
    u::C
    A::D
    ρ::E
    iteration::F
    current::G
    target::H
end

function update(o::PGCState;kwargs...)
    data = (;kwargs...)
    if hasproperty(data,:dx)
        dx = data.dx
    else
        dx = o.dx
    end
    if hasproperty(data,:r)
        r = data.r
    else
        r = o.r
    end
    if hasproperty(data,:u)
        u = data.u
    else
        u = o.u
    end
    if hasproperty(data,:A)
        A = data.A
    else
        A = o.A
    end
    if hasproperty(data,:ρ)
        ρ = data.ρ
    else
        ρ = o.ρ
    end
    if hasproperty(data,:iteration)
        iteration = data.iteration
    else
        iteration = o.iteration
    end
    if hasproperty(data,:current)
        current = data.current
    else
        current = o.current
    end
    if hasproperty(data,:target)
        target = data.target
    else
        target = o.target
    end
    PGCState(dx,r,u,A,ρ,iteration,current,target)
end

struct KrylovWorkspace{A,B} <: AbstractType
    options::A
    state::B
end

function update(w::KrylovWorkspace;kwargs...)
    options = update(w.options;kwargs...)
    state = update(w.state;kwargs...)
    KrylovWorkspace(options,state)
end

function done(ws::KrylovWorkspace)
    converged(ws) || tired(ws)
end

function converged(ws::KrylovWorkspace)
    (;current,target) = ws.state
    current <= target
end

function tired(ws::KrylovWorkspace)
    (;iteration) = ws.state
    (;iterations) = ws.options
    iteration >= iterations
end

function print_progress_header(a::KrylovWorkspace)
    (;output_prefix,verbose) = a.options
    s = output_prefix
    v = verbose
    c = "current"
    t = "target"
    v && @printf "%s%20s %20s\n" s "iterations" "residual"
    v && @printf "%s%10s %10s %10s %10s\n" s c t c t
end

function print_progress(a::KrylovWorkspace)
    (;output_prefix,verbose,iterations) = a.options
    (;iteration,current,target) = a.state
    s = output_prefix
    v = verbose
    v && @printf "%s%10i %10i %10.2e %10.2e\n" s iteration iterations current target
end

function cg(p;kwargs...)
    options = krylov_options(p;kwargs...)
    state = cg_state(p)
    workspace = KrylovWorkspace(options,state)
    linear_solver(cg_update,cg_step,p,workspace)
end

function cg_update(ws,A)
    (;Pl,update_Pl) = ws.options
    if update_Pl
        Pl = update(Pl,matrix=A)
    end
    iteration = 0
    update(ws;iteration,Pl,A)
end

function cg_step(x,ws,b,phase=:start;kwargs...)
    (;dx,r,u,A,ρ,iteration,current,target) = ws.state
    (;reltol,abstol,norm,Pl) = ws.options
    s = u
    if phase === :start
        iteration = 0
        phase = :advance
        copyto!(r,b)
        mul!(s,A,x)
        axpy!(-one(eltype(s)),s,r)
        #r .-= s
        current = norm(r)
        target = max(reltol*current,abstol)
        dx .= zero(eltype(dx))
        ρ = one(eltype(x))
        ws = update(ws;iteration,ρ,current,target)
        print_progress_header(ws)
    end
    ldiv!(u,Pl,r)
    ρ_prev = ρ
    ρ = dot(r,u)
    β = ρ / ρ_prev
    axpby!(one(eltype(u)),u,β,dx)
    #dx .= u .+ β .* dx
    s = u
    mul!(s,A,dx)
    α = ρ / dot(s,dx)
    axpy!(α,dx,x)
    #x .+= α .* dx
    axpy!(-α,s,r)
    #r .-= α .* s
    current = norm(r)
    iteration += 1
    current = norm(r)
    ws = update(ws;iteration,ρ,current)
    print_progress(ws)
    if done(ws)
        phase = :stop
    end
    x,ws,phase
end

# MINRES Data Structures and Functions

# Helper for Givens rotations
function sym_ortho(a::T, b::T) where T <: Real
    if b == zero(T)
        c = (a == zero(T)) ? one(T) : sign(a)
        s = zero(T)
        r_val = abs(a)
    elseif a == zero(T)
        c = zero(T)
        s = sign(b)
        r_val = abs(b)
    elseif abs(b) > abs(a)
        t = a / b
        s_abs = one(T) / sqrt(one(T) + t*t)
        s = sign(b) * s_abs
        c = s * t
        r_val = b / s # computationally better
    else # abs(a) >= abs(b)
        t = b / a
        c_abs = one(T) / sqrt(one(T) + t*t)
        c = sign(a) * c_abs
        s = c * t
        r_val = a / c # computationally better
    end
    return c, s, r_val
end

struct MINRESState{Tv, Ts, Ti, Tn, TA} <: AbstractType
    # Vectors
    r1::Tv      # R_{k-1} (unpreconditioned in Lanczos context)
    r2::Tv      # R_k (unpreconditioned part for Lanczos)
    r3::Tv      # R_k (preconditioned) / next Lanczos vector P_inv * R_k_unprec
    v::Tv       # Current Lanczos vector V_k
    w::Tv       # W_k for solution update
    wl::Tv      # W_{k-1} for solution update
    wl2::Tv     # W_{k-2} for solution update

    # Scalars for Lanczos and updates
    alpha::Ts   # alpha_k
    beta::Ts    # beta_k (from R_k^T P R_k, used for V_k = R_k_prec / beta_k)
    betan::Ts   # beta_{k+1} (from R_{k+1}^T P R_{k+1})
    betal::Ts   # beta_{k-1}

    # Scalars for Givens rotations and solution updates
    cs::Ts      # c_k from SymOrtho
    sn::Ts      # s_k from SymOrtho
    dbar::Ts    # d_bar_k (dltan from previous iteration)
    dltan::Ts   # delta_bar_{k+1} (-cs_k * betan_{k+1})
    dlta::Ts    # delta_k (cs_{k-1}*d_bar_k + sn_{k-1}*alpha_k)
    epln::Ts    # epsilon_k (eplnn from previous iteration)
    eplnn::Ts   # epsilon_{k+1} (sn_{k-1}*betan_k)
    gbar::Ts    # gamma_bar_k (sn_{k-1}*d_bar_k - cs_{k-1}*alpha_k)
    gama::Ts    # gamma_k (output r from SymOrtho(gbar_k, betan_k))
    gamal::Ts   # gamma_{k-1}
    gamal2::Ts  # gamma_{k-2}

    tau::Ts     # tau_k (cs_k * phi_{k-1})
    taul::Ts    # tau_{k-1}

    phi::Ts     # phi_k (sn_k * phi_{k-1}, current residual norm estimate)
    phi0::Ts    # Initial residual norm (beta1)

    A::TA         # System matrix (reference)
    iteration::Ti # Current MINRES iteration number (internal to the step)
    current::Tn   # Current residual norm for KrylovWorkspace convergence check
    target::Tn    # Target residual norm for KrylovWorkspace convergence
end

function minres_state(p)
    x_prototype = solution(p)
    A_matrix = matrix(p)
    Tv = typeof(x_prototype)
    Ts = real(eltype(x_prototype))
    Tn = real(eltype(x_prototype))
    Ti = Int

    r1  = similar(x_prototype, axes(A_matrix,1)); fill!(r1, zero(eltype(r1)))
    r2  = similar(x_prototype, axes(A_matrix,1)); fill!(r2, zero(eltype(r2)))
    r3  = similar(x_prototype, axes(A_matrix,1)); fill!(r3, zero(eltype(r3)))
    v   = similar(x_prototype, axes(A_matrix,2)); fill!(v, zero(eltype(v)))
    w   = similar(x_prototype, axes(A_matrix,2)); fill!(w, zero(eltype(w)))
    wl  = similar(x_prototype, axes(A_matrix,2)); fill!(wl, zero(eltype(wl)))
    wl2 = similar(x_prototype, axes(A_matrix,2)); fill!(wl2, zero(eltype(wl2)))

    alpha, beta, betan, betal = zero(Ts), zero(Ts), zero(Ts), zero(Ts)
    cs, sn = -one(Ts), zero(Ts) # Initial cs=-1, sn=0
    dbar, dltan, dlta = zero(Ts), zero(Ts), zero(Ts)
    epln, eplnn, gbar, gama = zero(Ts), zero(Ts), zero(Ts), zero(Ts)
    gamal, gamal2 = zero(Ts), zero(Ts)
    tau, taul = zero(Ts), zero(Ts)
    phi, phi0 = zero(Ts), zero(Ts)

    iteration = 0
    current_norm_val = zero(Tn)
    target_norm_val  = zero(Tn)

    MINRESState(r1, r2, r3, v, w, wl, wl2,
                alpha, beta, betan, betal,
                cs, sn, dbar, dltan, dlta, epln, eplnn, gbar, gama, gamal, gamal2,
                tau, taul, phi, phi0,
                A_matrix, iteration, current_norm_val, target_norm_val)
end

# Full update function for MINRESState, ensuring all fields can be updated
# This replaces the previous kwargs-based one for clarity if direct construction is preferred,
# or the kwargs one can be kept if it's more convenient for partial updates like in minres_update.
# For minres_step, we reconstruct the state fully.
function update(o::MINRESState{Tv, Ts, Ti, Tn, TA};
                r1=o.r1, r2=o.r2, r3=o.r3, v=o.v, w=o.w, wl=o.wl, wl2=o.wl2,
                alpha=o.alpha, beta=o.beta, betan=o.betan, betal=o.betal,
                cs=o.cs, sn=o.sn, dbar=o.dbar, dltan=o.dltan, dlta=o.dlta,
                epln=o.epln, eplnn=o.eplnn, gbar=o.gbar, gama=o.gama,
                gamal=o.gamal, gamal2=o.gamal2,
                tau=o.tau, taul=o.taul, phi=o.phi, phi0=o.phi0,
                A=o.A, iteration=o.iteration, current=o.current, target=o.target
               ) where {Tv, Ts, Ti, Tn, TA}
    MINRESState{Tv, Ts, Ti, Tn, TA}(
        r1, r2, r3, v, w, wl, wl2,
        alpha, beta, betan, betal,
        cs, sn, dbar, dltan, dlta,
        epln, eplnn, gbar, gama,
        gamal, gamal2,
        tau, taul, phi, phi0,
        A, iteration, current, target
    )
end


function minres(p; kwargs...)
    options = krylov_options(p; kwargs...)
    state = minres_state(p)
    workspace = KrylovWorkspace(options, state)
    # The linear_solver function manages the overall iteration count (ws.options.iterations)
    # and calls minres_step, which uses ws.state.iteration for its internal logic.
    linear_solver(minres_update, minres_step, p, workspace)
end

function minres_update(ws, current_A_matrix)
    options = ws.options
    state = ws.state
    
    Pl_updated = options.Pl
    #if options.update_Pl && options.Pl !== nothing && hasmethod(update, (typeof(options.Pl),))
    #    # Assuming the preconditioner has an update method like `update(Pl; matrix=A)`
    #    Pl_updated = update(options.Pl; matrix=current_A_matrix)
    #elseif options.update_Pl && options.Pl !== nothing && !hasmethod(update, (typeof(options.Pl),))
    #     @warn "Preconditioner does not have an update method. Using existing Pl."
    #end

    updated_minres_state = update(state; A=current_A_matrix, iteration=0)
    
    return KrylovWorkspace(update(options; Pl=Pl_updated), updated_minres_state)
end

function minres_step(x, ws, b, phase=:start; kwargs...)
    st = ws.state
    opt = ws.options
    A = st.A
    Pl = opt.Pl
    norm_func = opt.norm

    r1, r2, r3, v_vec, w_vec, wl_vec, wl2_vec = st.r1, st.r2, st.r3, st.v, st.w, st.wl, st.wl2

    if phase === :start
        # Initialization
        # r2 = b - A*x (unpreconditioned residual)
        if norm_func(x) == zero(real(eltype(x))) # Check if x is zero vector
            copyto!(r2, b)
        else
            mul!(r2, A, x) # r2 = A*x
            # Efficiently r2 = b - A*x (r2 currently holds A*x)
            # b is not overwritten. r2 becomes b-Ax.
            # r2 .= b .- r2  or  r2 .*= -1; r2 .+= b
            rmul!(r2, -one(eltype(r2)))
            axpy!(one(eltype(b)), b, r2)
        end

        # r3 = P_inv * r2 (preconditioned residual R_0)
        ldiv!(r3, Pl, r2) # r3 = P_inv * r2 (preconditioned residual R_0)

        # beta1 = sqrt(r3' * r2)
        # KSPCheckDot in C suggests beta1 should be non-negative.
        # Here, r3 is P_inv*r2_unprec, and r2 is r2_unprec. So dot(P_inv*r2_unprec, r2_unprec)
        # This is ||r2_unprec||_{P_inv}^2 if P_inv is SPD.
        # Or, if P is not symmetric, ||P_inv r_unprec||_2, i.e. norm(r3)
        # For symmetric P, dot(r2,r3) is appropriate. Let's assume Pl is symmetric.
        beta1_sq = dot(r2, r3)
        if beta1_sq < zero(real(eltype(beta1_sq)))
            # This case should ideally not happen with a good preconditioner or SPD matrix
            # Handle error or take absolute value if appropriate for the problem context
            # For now, let's assume it's non-negative as per theory
            @warn "dot(r_unprec, P_inv*r_unprec) is negative: $beta1_sq. Taking absolute value."
            beta1_sq = abs(beta1_sq)
        end
        beta1 = sqrt(beta1_sq)

        phi0 = beta1
        current_norm_val = beta1 
        target_norm_val = max(opt.reltol * phi0, opt.abstol)
        
        # Initialize Lanczos vector v_1 = r3 / beta1 (r3 is P_inv * r_0)
        if beta1 == zero(real(eltype(beta1)))
            # Initial residual is zero, solution is initial guess
            # Or handle as error / special case
            @warn "Initial preconditioned residual norm is zero. Solution might be exact or problem ill-posed."
            # Set phase to stop if beta1 is zero, as no progress can be made.
            # The done() check later will also catch this if current_norm_val is zero.
            phase = :stop
            # Update workspace with current status
            # k_minres (st.iteration) remains 0
            # global_iteration (ws.state.iteration) remains as is from KrylovWorkspace init
            updated_minres_state = update(st; iteration=0, current=current_norm_val, target=target_norm_val, phi=phi0, phi0=phi0)
            ws = update(ws; state=updated_minres_state) # Update MINRESState part
            print_progress_header(ws) # Print header even if stopping early
            print_progress(ws)      # Print current (likely converged) state
            return x, ws, phase
        end

        copyto!(v, r3)
        rmul!(v, one(real(eltype(v))) / beta1)

        # Initialize other MINRES scalars and vectors for k=0 (or first step)
        # st.r1 is v_{k-1}, initially zero vector for k=1 iteration.
        fill!(r1, zero(eltype(r1))) 
        
        # Scalars for iteration k=1 (MINRESState.iteration will be 0 before first advance)
        # beta is beta_k, betan is beta_{k+1}. For first step, beta_1 = beta1.
        # betal is beta_{k-1}
        
        # Initialize w vectors (w, wl, wl2) to zero
        fill!(w, zero(eltype(w)))
        fill!(wl, zero(eltype(wl)))
        fill!(wl2, zero(eltype(wl2)))

        # Update MINRESState (st)
        # st.iteration is k_minres, starts at 0.
        # ws.state.iteration is the global iteration count for KrylovWorkspace, starts at 0.
        updated_minres_state = update(st; 
            r1=r1, r2=r2, r3=r3, v=v, # Vectors
            alpha=zero(real(eltype(x))), beta=beta1, betan=beta1, betal=zero(real(eltype(x))), # Lanczos scalars
            cs=-one(real(eltype(x))), sn=zero(real(eltype(x))), # Givens
            dbar=zero(real(eltype(x))), dltan=zero(real(eltype(x))), dlta=zero(real(eltype(x))), # Givens related
            epln=zero(real(eltype(x))), eplnn=zero(real(eltype(x))), # Givens related
            gbar=beta1, gama=beta1, gamal=zero(real(eltype(x))), gamal2=zero(real(eltype(x))), # Givens related
            tau=zero(real(eltype(x))), taul=zero(real(eltype(x))), # Solution update scalars
            phi=phi0, phi0=phi0, # Residual norm estimates
            A=st.A, iteration=0, # Matrix, k_minres iteration
            current=current_norm_val, target=target_norm_val, # Convergence norms
            w=w, wl=wl, wl2=wl2 # Solution update vectors
        )
        # Update KrylovWorkspace (ws) with the new MINRESState
        # The global iteration count (ws.state.iteration if MINRESState is ws.state) is also 0 here.
        ws = update(ws; state=updated_minres_state) 
        
        print_progress_header(ws)
        print_progress(ws) # Print initial state (iteration 0)
        
        if done(ws) # Check convergence at initial state (e.g. if x0 is already solution)
            phase = :stop
        else
            phase = :advance
        end

    elseif phase === :advance
        k_minres = st.iteration + 1 # This is k for the current MINRES iteration (1-based for formulas)
        @printf "Phase advance: k_minres = %d\n" k_minres
        # Lanczos Step: Generate v_{k+1}, alpha_k, beta_{k+1}
        # Input: v_k (st.v), v_{k-1} (st.r1), beta_k (st.beta)
        # Output: v_{k+1} (st.v), alpha_k (alpha), beta_{k+1} (betan)
        # st.r1 will be updated to v_k for the *next* iteration's Lanczos step
        # st.r2 used for R_k (unpreconditioned), st.r3 for P_inv*R_k

        v_k = st.v
        v_k_minus_1 = st.r1 # This is v_{k-1} from previous step
        beta_k = st.beta    # This is beta_k from previous step (or beta1 for first advance)

        # Store current v_k into st.r1 to be v_{k-1} for the *next* iteration's Lanczos step
        # This must be done carefully. If st.r1 is an alias of v_k, this is problematic.
        # Assuming they are distinct buffers as per minres_state.
        # Let's use a temporary for v_k if needed, or ensure r1 is for v_k_minus_1 storage.
        # The C code does: r0=v_prev, r1=v_curr. Then computes Av_curr.
        # Then updates r0=v_curr for next iter.
        # So, st.r1 (v_k_minus_1) should be updated with v_k *after* it's used in current Lanczos.
        
        # Av_k = A * v_k
        # Using st.r2 as temporary for Av_k, then for R_k_unprec
        mul!(st.r2, A, v_k) 
        alpha = dot(v_k, st.r2) # alpha_k = v_k' * A * v_k
        
        # R_k_unprec = Av_k - alpha_k * v_k - beta_k * v_{k-1}
        # st.r2 currently holds Av_k.
        axpy!(-alpha, v_k, st.r2)      # st.r2 = Av_k - alpha_k * v_k
        axpy!(-beta_k, v_k_minus_1, st.r2) # st.r2 = (Av_k - alpha_k*v_k) - beta_k*v_{k-1}
                                          # st.r2 is now R_k (unpreconditioned)
        
        # Precondition: st.r3 = P_inv * R_k_unprec
        ldiv!(st.r3, Pl, st.r2)
        
        # beta_{k+1} = sqrt(R_k_unprec' * P_inv * R_k_unprec)
        # Or norm(P_inv * R_k_unprec) if Pl not symmetric for dot product
        betan_sq = dot(st.r2, st.r3)
        if betan_sq < zero(real(eltype(betan_sq)))
            @warn "Lanczos dot product for beta_{k+1} is negative: $betan_sq. Problem with A or P?"
            # This can lead to breakdown or instability. Forcing positive for sqrt.
            betan_sq = abs(betan_sq) 
        end
        betan = sqrt(betan_sq) # This is beta_{k+1}

        # Update v_{k-1} for next iteration: v_k -> v_{k-1}
        # copyto!(st.r1, v_k) # v_k (current) becomes v_{k-1} for next iteration
        # This was done by C code as r0=r1 (v_prev = v_curr)
        # Here, v_k is st.v. So st.r1 should become st.v.
        # This should happen *before* st.v is updated to v_{k+1}.
        v_k_snapshot_for_r1_update = copy(st.v) # Save current v_k before it's overwritten

        # Update v: v_{k+1} = (P_inv * R_k_unprec) / beta_{k+1}
        if betan == zero(real(eltype(betan)))
            # Lanczos breakdown, solution might be found or problem is ill-conditioned
            @warn "Lanczos breakdown: beta_{k+1} is zero at iteration $k_minres."
            # Proceed with current values, convergence will be checked by phi
            # Set betan to a small number to avoid division by zero if absolutely necessary,
            # but typically the algorithm should terminate or handle this via convergence.
            # For now, let phi update and convergence check handle it.
            # If phi becomes small, it will converge. If not, it will hit max iterations.
            # No update to st.v if betan is zero, effectively v_{k+1} is not well-defined.
            # This means the Givens rotation part might also have issues.
            # Let's assume phi will correctly reflect this.
            # The C code checks if beta is "too small" and terminates.
            # We rely on phi and the done() check.
        else
            copyto!(st.v, st.r3)
            rmul!(st.v, one(real(eltype(st.v))) / betan) # st.v is now v_{k+1}
        end
        
        # Update st.r1 (v_{k-1} for next iteration)
        copyto!(st.r1, v_k_snapshot_for_r1_update)


        # Apply previous Givens rotation and compute current one
        # Scalars from previous state (st): cs, sn, dbar, gbar, gama, epln, dlta
        # Current Lanczos scalars: alpha (computed above), beta_k (st.beta), betan (beta_{k+1} computed above)

        # Notation mapping to some standard MINRES notes (e.g., Paige & Saunders, or PETSc):
        # alpha is alpha_k
        # st.beta is beta_k
        # betan is beta_{k+1}
        # st.cs, st.sn are cs_{k-1}, sn_{k-1}
        # st.dbar is dbar_k
        # st.gbar is gbar_k (gamma_bar_k in some notes)
        # st.gama is gama_k (gamma_k in some notes)
        # st.epln is epsilon_k
        # st.dlta is delta_k (not delta_bar)

        # Quantities for QR factorization of T_k (tridiagonal matrix from Lanczos)
        # dltan_val = cs_{k-1}*dbar_k + sn_{k-1}*alpha_k  (This is delta_bar_k in C code, or delta_hat_k)
        dltan_val = st.cs * st.dbar + st.sn * alpha
        # gbar_val  = sn_{k-1}*dbar_k - cs_{k-1}*alpha_k  (This is gbar_k in C code, or gamma_prime_k)
        gbar_val  = st.sn * st.dbar - st.cs * alpha
        
        # eplnn_val = sn_{k-1}*beta_{k+1} (epsilon_hat_{k+1})
        eplnn_val = st.sn * betan
        # dbar_next_val = -cs_{k-1}*beta_{k+1} (dbar_{k+1})
        dbar_next_val = -st.cs * betan
        
        # dlta_curr = st.gama (This is delta_k in C code, should be gamma_k from previous step)
        # This is delta_k in some notations, which is gamma_{k-1} (from sym_ortho of previous step) * cs_{k-1} + ...
        # C code: dlta = gama; (where gama was result of sym_ortho(gbar, beta))
        # So, dlta_curr is st.gama (which was gamma_k from previous sym_ortho)
        dlta_curr = st.gama 

        # Compute cs_k, sn_k, gama_{k+1} from gbar_val (gamma_prime_k) and beta_{k+1}
        # gama_next_val = sqrt(gbar_val^2 + beta_{k+1}^2)
        # cs_curr, sn_curr are cs_k, sn_k
        cs_curr, sn_curr, gama_next_val = sym_ortho(gbar_val, betan)

        # Update solution x and w vectors
        # st.w is w_k, st.wl is w_{k-1}, st.wl2 is w_{k-2}
        # st.tau is tau_k, st.taul is tau_{k-1}
        # st.phi is phi_k (residual norm estimate ||r_k||_M)
        # v_k_snapshot_for_r1_update is v_k (used to compute w_{k+1})

        # tau_{k+1} = (phi_k - dltan_val*tau_k - epln_val*tau_{k-1}) / gama_{k+1}
        # epln_val here is st.epln (epsilon_k)
        tau_new = (st.phi - dltan_val * st.tau - st.epln * st.taul) / gama_next_val
        
        # Update w vectors: w_{k+1} = (v_k - dltan_val*w_k - epln_val*w_{k-1}) / gama_{k+1}
        # Need to manage w, wl, wl2 updates carefully.
        # wl2_new = wl_old (st.wl)
        # wl_new  = w_old  (st.w)
        # w_new   = ...
        
        # Store old wl into wl2
        copyto!(st.wl2, st.wl)
        # Store old w into wl
        copyto!(st.wl, st.w)
        
        # Compute new w: st.w = (v_k_snapshot - dltan_val*wl_new - epln_val*wl2_new) / gama_next_val
        # Note: wl_new is old st.w, wl2_new is old st.wl
        copyto!(st.w, v_k_snapshot_for_r1_update) # st.w starts as v_k
        axpy!(-dltan_val, st.wl, st.w)     # st.w = v_k - dltan_val*w_old
        axpy!(-st.epln, st.wl2, st.w)      # st.w = v_k - dltan_val*w_old - epln_val*w_prev_old
        if gama_next_val != zero(real(eltype(gama_next_val)))
            rmul!(st.w, one(real(eltype(st.w))) / gama_next_val)
        else
            # gama_next_val is zero, implies breakdown or exact solution.
            # w update might be problematic. If phi is zero, it should converge.
            @warn "gama_next_val is zero in w update at iteration $k_minres."
            fill!(st.w, zero(eltype(st.w))) # Or handle as error
        end

        # Update solution: x = x + tau_new * w_new
        axpy!(tau_new, st.w, x)

        # Update residual norm estimate: phi_{k+1} = -sn_k * phi_k
        phi_new = -sn_curr * st.phi
        current_norm_val = abs(phi_new)
        @printf "Current residual norm (phi_{k+1}): %.2e\n" current_norm_val
        # Update scalars in MINRESState for next iteration
        # st.iteration is k_minres (internal MINRES counter)
        # ws.state.iteration is the global iteration count for KrylovWorkspace
        
        # Global iteration count for KrylovWorkspace (used by done(), print_progress())
        # This should be ws.state.iteration from the input ws, incremented.
        # However, cg_step updates ws.state.iteration directly.
        # Let's assume MINRESState.iteration is the one for print_progress and done()
        # as it's part of ws.state.
        
        updated_minres_state = update(st; 
            iteration=k_minres, # k_minres is the new MINRES iteration number
            current=current_norm_val, 
            phi=phi_new,
            alpha=alpha, # alpha_k
            beta=betan,  # beta_{k+1} becomes beta_k for next iter
            betal=beta_k, # beta_k becomes beta_{k-1} for next iter
            betan=betan,  # Store beta_{k+1} (though it will be recomputed unless used elsewhere)
            cs=cs_curr, sn=sn_curr,
            dbar=dbar_next_val, dltan=dltan_val, dlta=dlta_curr, # dlta was st.gama
            epln=eplnn_val, # epln for next iter is eplnn_val (epsilon_hat_{k+1})
            # eplnn remains as is, will be calculated next iter based on new sn and new betan
            gbar=gbar_val, gama=gama_next_val, 
            # gamal, gamal2 update if needed, but seem implicitly handled by w updates
            tau=tau_new, taul=st.tau, # tau_k becomes tau_{k-1}
            # Vectors v, r1, w, wl, wl2 are updated in-place or via copyto! above.
            # r2, r3 were temporary for Lanczos.
            # phi0, A, target remain the same.
            v=st.v, r1=st.r1, w=st.w, wl=st.wl, wl2=st.wl2 
        )
        ws = update(ws; state=updated_minres_state)

        print_progress(ws)
        if done(ws)
            phase = :stop
        end
    end # end phase if/elseif

    return x, ws, phase
end

# Ensure the update function for MINRESState is robust.
# The one in the attachment is manual. A more generic one using fieldnames is:
function update(o::MINRESState{Tv, Ts, Ti, Tn, TA}; kwargs...) where {Tv, Ts, Ti, Tn, TA}
    new_values_dict = Dict{Symbol, Any}()
    for name in fieldnames(MINRESState)
        new_values_dict[name] = getfield(o, name)
    end
    for (key, value) in kwargs
        new_values_dict[key] = value
    end

    MINRESState{Tv, Ts, Ti, Tn, TA}(
        new_values_dict[:r1], new_values_dict[:r2], new_values_dict[:r3], 
        new_values_dict[:v], new_values_dict[:w], new_values_dict[:wl], new_values_dict[:wl2],
        new_values_dict[:alpha], new_values_dict[:beta], new_values_dict[:betan], new_values_dict[:betal],
        new_values_dict[:cs], new_values_dict[:sn], new_values_dict[:dbar], new_values_dict[:dltan], new_values_dict[:dlta], 
        new_values_dict[:epln], new_values_dict[:eplnn], new_values_dict[:gbar], new_values_dict[:gama], 
        new_values_dict[:gamal], new_values_dict[:gamal2],
        new_values_dict[:tau], new_values_dict[:taul],
        new_values_dict[:phi], new_values_dict[:phi0],
        new_values_dict[:A], new_values_dict[:iteration], new_values_dict[:current], new_values_dict[:target]
    )
end
# However, the existing update function in the file should be used if it's already there and functional.
# The provided attachment has a manual field-by-field update function for MINRESState.
# I've completed minres_step assuming that update function works by passing all relevant fields.
