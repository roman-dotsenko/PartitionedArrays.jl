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
    @show current, target
    current <= target
end

function tired(ws::KrylovWorkspace)
    (;iteration) = ws.state
    (;iterations) = ws.options
    @show iteration, iterations
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
    gamal3::Ts  # gamma_{k-3}

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
    gamal, gamal2, gamal3 = zero(Ts), zero(Ts), zero(Ts)
    tau, taul = zero(Ts), zero(Ts)
    phi, phi0 = zero(Ts), zero(Ts)

    iteration = 0
    current_norm_val = zero(Tn)
    target_norm_val  = zero(Tn)

    MINRESState(r1, r2, r3, v, w, wl, wl2,
                alpha, beta, betan, betal,
                cs, sn, dbar, dltan, dlta, epln, eplnn, gbar, gama, gamal, gamal2, gamal3,
                tau, taul, phi, phi0,
                A_matrix, iteration, current_norm_val, target_norm_val)
end

function update(o::MINRESState;kwargs...)
    data = (;kwargs...)
    if hasproperty(data,:r1)
        r1 = data.r1
    else
        r1 = o.r1
    end
    if hasproperty(data,:r2)
        r2 = data.r2
    else
        r2 = o.r2
    end
    if hasproperty(data,:r3)
        r3 = data.r3
    else
        r3 = o.r3
    end
    if hasproperty(data,:v)
        v = data.v
    else
        v = o.v
    end
    if hasproperty(data,:w)
        w = data.w
    else
        w = o.w
    end
    if hasproperty(data,:wl)
        wl = data.wl
    else
        wl = o.wl
    end
    if hasproperty(data,:wl2)
        wl2 = data.wl2
    else
        wl2 = o.wl2
    end
    if hasproperty(data,:alpha)
        alpha = data.alpha
    else
        alpha = o.alpha
    end
    if hasproperty(data,:beta)
        beta = data.beta
    else
        beta = o.beta
    end
    if hasproperty(data,:betan)
        betan = data.betan
    else
        betan = o.betan
    end
    if hasproperty(data,:betal)
        betal = data.betal
    else
        betal = o.betal
    end
    if hasproperty(data,:cs)
        cs = data.cs
    else
        cs = o.cs
    end
    if hasproperty(data,:sn)
        sn = data.sn
    else
        sn = o.sn
    end
    if hasproperty(data,:dbar)
        dbar = data.dbar
    else
        dbar = o.dbar
    end
    if hasproperty(data,:dltan)
        dltan = data.dltan
    else
        dltan = o.dltan
    end
    if hasproperty(data,:dlta)
        dlta = data.dlta
    else
        dlta = o.dlta
    end
    if hasproperty(data,:epln)
        epln = data.epln
    else
        epln = o.epln
    end
    if hasproperty(data,:eplnn)
        eplnn = data.eplnn
    else
        eplnn = o.eplnn
    end
    if hasproperty(data,:gbar)
        gbar = data.gbar
    else
        gbar = o.gbar
    end
    if hasproperty(data,:gama)
        gama = data.gama
    else
        gama = o.gama
    end
    if hasproperty(data,:gamal)
        gamal = data.gamal
    else
        gamal = o.gamal
    end
    if hasproperty(data,:gamal2)
        gamal2 = data.gamal2
    else
        gamal2 = o.gamal2
    end
    if hasproperty(data,:gamal3)
        gamal3 = data.gamal3
    else
        gamal3 = o.gamal3
    end
    if hasproperty(data,:tau)
        tau = data.tau
    else
        tau = o.tau
    end
    if hasproperty(data,:taul)
        taul = data.taul
    else
        taul = o.taul
    end
    if hasproperty(data,:phi)
        phi = data.phi
    else
        phi = o.phi
    end
    if hasproperty(data,:phi0)
        phi0 = data.phi0
    else
        phi0 = o.phi0
    end
    if hasproperty(data,:A)
        A = data.A
    else
        A = o.A
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
    MINRESState(r1,r2,r3,v,w,wl,wl2,alpha,beta,betan,betal,cs,sn,dbar,dltan,dlta,epln,eplnn,gbar,gama,gamal,gamal2,gamal3,tau,taul,phi,phi0,A,iteration,current,target)
end

function minres(p; kwargs...)
    options = krylov_options(p; kwargs...)
    state = minres_state(p)
    workspace = KrylovWorkspace(options, state)
    linear_solver(minres_update, minres_step, p, workspace)
end

function minres_update(ws, current_A_matrix)
    (;Pl,update_Pl) = ws.options
    if update_Pl
        Pl = update(Pl,matrix=current_A_matrix)
    end
    iteration = 0
    update(ws;iteration,Pl,A=current_A_matrix)
end

function minres_step(x, ws, b, phase=:start; kwargs...)
    st = ws.state
    opt = ws.options
    A = st.A
    Pl = opt.Pl
    norm_func = opt.norm

    r1, r2, r3, v, w, wl, wl2 = st.r1, st.r2, st.r3, st.v, st.w, st.wl, st.wl2

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
        
        # Initialize missing variables from C code (lines 162-165)
        relres = current_norm_val / beta1  # This should be 1.0 initially
        betan = beta1
        phi = beta1
        beta = zero(real(eltype(x)))
        
        # Initialize Lanczos vector v_1 = r3 / beta1 (r3 is P_inv * r_0)
        if beta1 == zero(real(eltype(beta1)))
            # Initial residual is zero, solution is initial guess
            # Or handle as error / special case
            @warn "Initial preconditioned residual norm is zero. Solution might be exact or problem ill-posed."
            # Set phase to stop if beta1 is zero, as no progress can be made.
            # The done() check later will also catch this if current_norm_val is zero.
            phase = :stop            # Update workspace with current status following CG pattern
            iteration = 0
            current = current_norm_val
            target = target_norm_val
            phi = phi0
            ws = update(ws;iteration,current,target,phi,phi0)
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
        fill!(wl2, zero(eltype(wl2)))        # Update scalar values in MINRESState following CG pattern
        # Vectors have been modified in-place above, only pass scalar values
        alpha = zero(real(eltype(x)))
        beta = beta1
        betan = beta1
        betal = zero(real(eltype(x)))
        cs = -one(real(eltype(x)))
        sn = zero(real(eltype(x)))
        dbar = zero(real(eltype(x)))
        dltan = zero(real(eltype(x)))
        dlta = zero(real(eltype(x)))
        epln = zero(real(eltype(x)))
        eplnn = zero(real(eltype(x)))
        gbar = beta1
        gama = beta1
        gamal = zero(real(eltype(x)))
        gamal2 = zero(real(eltype(x)))
        tau = zero(real(eltype(x)))
        taul = zero(real(eltype(x)))
        phi = phi0
        iteration = 1
        current = current_norm_val
        target = target_norm_val
        ws = update(ws;alpha,beta,betan,betal,cs,sn,dbar,dltan,dlta,epln,eplnn,gbar,gama,gamal,gamal2,tau,taul,phi,phi0,iteration,current,target)
        
        print_progress_header(ws)
        print_progress(ws) # Print initial state (iteration 0)
        if done(ws) # Check convergence at initial state (e.g. if x0 is already solution)
            phase = :stop
        else
            phase = :advance
        end
    elseif phase === :advance
        k_minres = st.iteration + 1 # This is k for the current MINRES iteration (1-based for formulas)
        @printf "Phase advance: k_minres = %d\n" k_minres        # Lanczos Step: Generate v_{k+1}, alpha_k, beta_{k+1}
        # Following C code lines 172-189
          # Lanczos Step: betal = beta; beta = betan (C lines 173-174)
        betal_new = st.beta
        beta_new = st.betan
        
        # v = r3 / beta (C line 175)
        copyto!(st.v, st.r3)
        rmul!(st.v, one(real(eltype(st.v))) / beta_new)
        
        # r3 = A * v (C line 176)
        mul!(st.r3, A, st.v)
        
        # r3 = r3 - (beta/betal) * r1 if k > 1 (C lines 177-178)
        if k_minres > 1
            axpy!(-beta_new / betal_new, st.r1, st.r3)
        end
        
        # alpha = dot(r3, v) (C line 179)
        alpha_new = dot(st.r3, st.v)
        
        # r3 = r3 - (alpha/beta) * r2 (C line 180)
        axpy!(-alpha_new / beta_new, st.r2, st.r3)
        
        # KSPMinresSwap3(R1, R2, R3) - critical vector swapping (C line 181)
        # This swaps R1 ← R2, R2 ← R3, R3 ← R1 in the C code
        temp_r1 = copy(st.r1)
        copyto!(st.r1, st.r2)
        copyto!(st.r2, st.r3)
        copyto!(st.r3, temp_r1)
          # Precondition: st.r3 = P_inv * R_k_unprec (r2 is now the unpreconditioned residual)
        ldiv!(st.r3, Pl, st.r2)
        
        # beta_{k+1} = sqrt(R_k_unprec' * P_inv * R_k_unprec)
        betan_sq = dot(st.r2, st.r3)
        if betan_sq < zero(real(eltype(betan_sq)))
            @warn "Lanczos dot product for beta_{k+1} is negative: $betan_sq"
            betan_sq = abs(betan_sq) 
        end
        betan_new = sqrt(betan_sq)

        # Apply previous left rotation Q_{k-1} (C lines 203-208)
        # dbar = dltan, epln = eplnn from previous iteration
        dbar_new = st.dltan
        epln_new = st.eplnn
        dlta_new = st.cs * dbar_new + st.sn * alpha_new
        gbar_new = st.sn * dbar_new - st.cs * alpha_new
        eplnn_new = st.sn * betan_new
        dltan_new = -st.cs * betan_new        # Compute current left plane rotation Q_k (C lines 230-233)
        gamal3_new = st.gamal2
        gamal2_new = st.gamal
        gamal_new = st.gama
        cs_new, sn_new, gama_new = sym_ortho(gbar_new, betan_new)

        # Update tau and phi using CURRENT rotation (C lines 252-255)
        # CRITICAL: Use cs_new, sn_new (current rotation), not st.cs, st.sn (previous)
        tau_new = cs_new * st.phi
        phi_new = sn_new * st.phi        # Update w vectors and solution x (C lines 300-309)
        # Use gama_tmp (previous gama) for w update, not gama_new
        gama_tmp = st.gama  # This is gama from previous iteration
        
        # Perform vector swapping: WL2 ← WL, WL ← W (C line 300)
        copyto!(st.wl2, st.wl)
        copyto!(st.wl, st.w)
        
        # Compute new w: W = V / gama_tmp (C line 301)
        copyto!(st.w, st.v)
        if gama_tmp != zero(real(eltype(gama_tmp)))
            rmul!(st.w, one(real(eltype(st.w))) / gama_tmp)
        else
            @warn "gama_tmp is zero in w update at iteration $k_minres"
            fill!(st.w, zero(eltype(st.w)))
        end
          # Apply corrections if iterations > 1 (C lines 302-307)
        # Use dlta_new and st.eplnn (previous eplnn) with gama_tmp
        if k_minres > 1
            nv = (k_minres == 2) ? 1 : 2
            if nv >= 1
                axpy!(-dlta_new / gama_tmp, st.wl, st.w)
            end
            if nv >= 2
                axpy!(-st.eplnn / gama_tmp, st.wl2, st.w)  # Use st.eplnn, not eplnn_new
            end
        end

        # Update solution: x = x + tau*w (C line 309)
        axpy!(tau_new, st.w, x)

        current_norm_val = abs(phi_new)
        @printf "Current residual norm (phi_{k+1}): %.2e\n" current_norm_val

        # Update all scalar values in MINRESState following CG pattern
        # Vectors have been modified in-place above, only pass scalar values
        iteration = k_minres
        current = current_norm_val
        alpha = alpha_new
        beta = betan_new  # beta_{k+1} becomes beta_k for next iter
        betal = beta_new  # beta_k becomes beta_{k-1} for next iter
        betan = betan_new # This stays as beta_{k+1}
        cs = cs_new
        sn = sn_new
        dbar = dbar_new
        dltan = dltan_new
        dlta = dlta_new
        epln = epln_new
        eplnn = eplnn_new
        gbar = gbar_new
        gama = gama_new
        gamal = gamal_new
        gamal2 = gamal2_new
        gamal3 = gamal3_new
        tau = tau_new
        taul = st.tau # tau_k becomes tau_{k-1}
        phi = phi_new
        ws = update(ws;iteration,current,phi,alpha,beta,betal,betan,cs,sn,dbar,dltan,dlta,epln,eplnn,gbar,gama,gamal,gamal2,gamal3,tau,taul)

        print_progress(ws)
        if done(ws)
            phase = :stop
        end
    end # end phase if/elseif

    return x, ws, phase
end
