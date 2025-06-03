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

# Helper function for Givens rotations (symmetric case)
function sym_givens(a, b)
    T = promote_type(typeof(a), typeof(b))
    if b == zero(T)
        if a == zero(T)
            c = one(T)
        else
            c = sign(a)
        end
        s = zero(T)
        d = abs(a)
    elseif a == zero(T)
        c = zero(T)
        s = sign(b)
        d = abs(b)
    elseif abs(b) > abs(a)
        t = a/b
        s = sign(b) / sqrt(one(T) + t^2)
        c = s*t
        d = b/s
    else
        t = b/a
        c = sign(a) / sqrt(one(T) + t^2)
        s = c*t
        d = a/c
    end
    return c, s, d
end

function minres_options(p;
        shift = zero(real(eltype(rhs(p)))),
        maxxnorm = 1e7,
        acondlim = 1e15,
        trancond = 1e7,
        kwargs...
    )
    base_options = krylov_options(p; kwargs...)
    MinresOptions(
        base_options.iterations, base_options.abstol, base_options.reltol, 
        base_options.norm, base_options.Pl, base_options.update_Pl, 
        base_options.verbose, base_options.output_prefix,
        shift, maxxnorm, acondlim, trancond
    )
end

struct MinresOptions{A,B,C,D,E,F,G,H,I,J,K,L} <: AbstractType
    # KrylovOptions fields
    iterations::A
    abstol::B
    reltol::C
    norm::D
    Pl::E
    update_Pl::F
    verbose::G
    output_prefix::H
    # MINRES-specific fields
    shift::I
    maxxnorm::J
    acondlim::K
    trancond::L
end

function update(o::MinresOptions; kwargs...)
    data = (;kwargs...)
    
    # Handle KrylovOptions fields
    iterations = hasproperty(data, :iterations) ? data.iterations : o.iterations
    abstol = hasproperty(data, :abstol) ? data.abstol : o.abstol
    reltol = hasproperty(data, :reltol) ? data.reltol : o.reltol
    norm = hasproperty(data, :norm) ? data.norm : o.norm
    Pl = hasproperty(data, :Pl) ? data.Pl : o.Pl
    update_Pl = hasproperty(data, :update_Pl) ? data.update_Pl : o.update_Pl
    verbose = hasproperty(data, :verbose) ? data.verbose : o.verbose
    output_prefix = hasproperty(data, :output_prefix) ? data.output_prefix : o.output_prefix
    
    # Handle MINRES-specific fields
    shift = hasproperty(data, :shift) ? data.shift : o.shift
    maxxnorm = hasproperty(data, :maxxnorm) ? data.maxxnorm : o.maxxnorm
    acondlim = hasproperty(data, :acondlim) ? data.acondlim : o.acondlim
    trancond = hasproperty(data, :trancond) ? data.trancond : o.trancond
    
    MinresOptions(iterations, abstol, reltol, norm, Pl, update_Pl, verbose, output_prefix,
                  shift, maxxnorm, acondlim, trancond)
end

function minres_state(p)
    x = solution(p)
    A = matrix(p)
    T = real(eltype(x))
    
    # Lanczos vectors
    v = similar(x, axes(A, 2))
    r1 = similar(x, axes(A, 1))
    r2 = similar(x, axes(A, 1))
    r3 = similar(x, axes(A, 1))
    
    # Direction vectors
    w = similar(x, axes(A, 2))
    wl = similar(x, axes(A, 2))
    wl2 = similar(x, axes(A, 2))
      # Scalar quantities
    beta = zero(T)
    betal = zero(T)
    betan = zero(T)
    beta1 = zero(T)
    alfa = zero(T)
    
    # Rotation parameters
    cs = -one(T)
    sn = zero(T)
    cr1 = -one(T)
    sr1 = zero(T)
    cr2 = -one(T)
    sr2 = zero(T)
    
    # QLP quantities
    dltan = zero(T)
    eplnn = zero(T)
    gama = zero(T)
    gamal = zero(T)
    gamal2 = zero(T)
    eta = zero(T)
    etal = zero(T)
    etal2 = zero(T)
    vepln = zero(T)
    veplnl = zero(T)
    veplnl2 = zero(T)
    
    # Solution norm tracking
    ul = zero(T)
    ul2 = zero(T)
    ul3 = zero(T)
    ul4 = zero(T)
    u = zero(T)
    xnorm = zero(T)
    xl2norm = zero(T)
    
    # Matrix estimates
    anorm = zero(T)
    acond = one(T)
    gmin = zero(T)
    
    # Residual tracking
    phi = zero(T)
    tau = zero(T)
    taul = zero(T)
    taul2 = zero(T)
    
    # Flags and counters
    qlpiter = 0
    minres_mode = true
    
    # Storage for QLP transition
    gamal_qlp = zero(T)
    vepln_qlp = zero(T)
    gama_qlp = zero(T)
    ul_qlp = zero(T)
    u_qlp = zero(T)
    iteration = 0
    current = zero(T)
    target = zero(T)
    
    MinresState(v, r1, r2, r3, w, wl, wl2, A, 
                beta, betal, betan, beta1, alfa,
                cs, sn, cr1, sr1, cr2, sr2,
                dltan, eplnn, gama, gamal, gamal2,
                eta, etal, etal2, vepln, veplnl, veplnl2,
                ul, ul2, ul3, ul4, u, xnorm, xl2norm,
                anorm, acond, gmin, phi, tau, taul, taul2,
                qlpiter, minres_mode,
                gamal_qlp, vepln_qlp, gama_qlp, ul_qlp, u_qlp,
                iteration, current, target)
end

struct MinresState{V1,V2,V3,V4,V5,V6,V7,M,T} <: AbstractType
    v::V1
    r1::V2
    r2::V3
    r3::V4
    w::V5
    wl::V6
    wl2::V7
    A::M
      # Lanczos scalars
    beta::T
    betal::T
    betan::T
    beta1::T  # Initial residual norm
    alfa::T
    
    # Rotation parameters
    cs::T
    sn::T
    cr1::T
    sr1::T
    cr2::T
    sr2::T
    
    # QLP quantities
    dltan::T
    eplnn::T
    gama::T
    gamal::T
    gamal2::T
    eta::T
    etal::T
    etal2::T
    vepln::T
    veplnl::T
    veplnl2::T
    
    # Solution norm tracking
    ul::T
    ul2::T
    ul3::T
    ul4::T
    u::T
    xnorm::T
    xl2norm::T
    
    # Matrix estimates
    anorm::T
    acond::T
    gmin::T
    
    # Residual tracking
    phi::T
    tau::T
    taul::T
    taul2::T
    
    # Flags and counters
    qlpiter::Int
    minres_mode::Bool
    
    # Storage for QLP transition
    gamal_qlp::T
    vepln_qlp::T
    gama_qlp::T
    ul_qlp::T
    u_qlp::T
    
    # Common iteration tracking
    iteration::Int
    current::T
    target::T
end

function update(o::MinresState; kwargs...)
    data = (;kwargs...)
    
    # Helper macro to update fields
    function get_field(field_name)
        if hasproperty(data, field_name)
            return getproperty(data, field_name)
        else
            return getproperty(o, field_name)
        end
    end
      MinresState(
        get_field(:v), get_field(:r1), get_field(:r2), get_field(:r3),
        get_field(:w), get_field(:wl), get_field(:wl2), get_field(:A),
        get_field(:beta), get_field(:betal), get_field(:betan), get_field(:beta1), get_field(:alfa),
        get_field(:cs), get_field(:sn), get_field(:cr1), get_field(:sr1),
        get_field(:cr2), get_field(:sr2), get_field(:dltan), get_field(:eplnn),
        get_field(:gama), get_field(:gamal), get_field(:gamal2),
        get_field(:eta), get_field(:etal), get_field(:etal2),
        get_field(:vepln), get_field(:veplnl), get_field(:veplnl2),
        get_field(:ul), get_field(:ul2), get_field(:ul3), get_field(:ul4),
        get_field(:u), get_field(:xnorm), get_field(:xl2norm),
        get_field(:anorm), get_field(:acond), get_field(:gmin),
        get_field(:phi), get_field(:tau), get_field(:taul), get_field(:taul2),
        get_field(:qlpiter), get_field(:minres_mode),
        get_field(:gamal_qlp), get_field(:vepln_qlp), get_field(:gama_qlp),
        get_field(:ul_qlp), get_field(:u_qlp),
        get_field(:iteration), get_field(:current), get_field(:target)
    )
end

function minres(p; kwargs...)
    options = minres_options(p; kwargs...)
    state = minres_state(p)
    workspace = KrylovWorkspace(options, state)
    linear_solver(minres_update, minres_step, p, workspace)
end

function minres_update(ws, A)
    (;Pl, update_Pl) = ws.options
    if update_Pl
        Pl = update(Pl, matrix=A)
    end
    iteration = 0
    options_updated = update(ws.options; iteration, Pl, A)
    state_updated = update(ws.state; iteration, A)
    update(ws; options=options_updated, state=state_updated)
end

function minres_step(x, ws, b, phase=:start; kwargs...)
    state = ws.state
    options = ws.options
    
    (;reltol, abstol, norm, Pl) = options
    (;shift, maxxnorm, acondlim, trancond) = options
    
    if phase === :start
        return minres_start(x, ws, b, norm, reltol, abstol, Pl, shift)
    else
        return minres_iterate(x, ws, b, norm, Pl, shift, maxxnorm, acondlim, trancond)
    end
end

function minres_start(x, ws, b, norm, reltol, abstol, Pl, shift)
    state = ws.state
    (;r1, r2, r3, A) = state
    
    # Initial setup
    iteration = 0
    phase = :advance
    
    # Compute initial residual r = b - A*x (preserve initial guess)
    mul!(r2, A, x)      # r2 = A*x
    copyto!(r3, b)      # r3 = b  
    axpy!(-one(eltype(r3)), r2, r3)  # r3 = b - A*x
    copyto!(r2, r3)     # r2 = b - A*x
    
    # Compute initial beta (with preconditioning if applicable)
    if Pl !== nothing
        ldiv!(r3, Pl, r2)  # r3 = M \ r2
        beta1 = sqrt(real(dot(r2, r3)))
        if beta1 < 0
            error("Preconditioner appears to be indefinite")
        end
    else
        beta1 = norm(r2)
    end
    
    # Initialize residual tracking
    current = beta1
    target = max(reltol * current, abstol)
    
    # Initialize scalar quantities
    betan = beta1
    phi = beta1
    tau = zero(eltype(x))
    cs = -one(real(eltype(x)))
    sn = zero(real(eltype(x)))
    
    # Initialize working vectors
    fill!(state.w, zero(eltype(x)))
    fill!(state.wl, zero(eltype(x)))
    fill!(state.wl2, zero(eltype(x)))
    
    # Check for trivial case
    if beta1 == 0
        phase = :stop
    end
      # Update state
    ws_updated = update(ws; 
        iteration, current, target, 
        betan, beta1, phi, tau, cs, sn,
        minres_mode=true, qlpiter=0
    )
    
    print_progress_header(ws_updated)
    
    return x, ws_updated, phase
end

function minres_iterate(x, ws, b, norm, Pl, shift, maxxnorm, acondlim, trancond)
    state = ws.state
    (;v, r1, r2, r3, w, wl, wl2, A) = state
    (;beta, betan, phi, tau, cs, sn, iteration) = state
    (;minres_mode, qlpiter) = state
      # Update iteration counter
    iteration += 1
    
    # Lanczos step
    betal = beta
    beta = betan
    
    # v = r3 / beta
    v .= r3 ./ beta
    
    # r3 = A * v
    mul!(r3, A, v)
    
    # Apply shift if needed
    if shift != 0
        axpy!(-shift, v, r3)  # r3 = r3 - shift * v
    end
    
    # Three-term recurrence
    if iteration > 1
        axpy!(-beta/betal, r1, r3)  # r3 = r3 - (beta/betal) * r1
    end
    
    # Compute diagonal element
    alfa = real(dot(r3, v))
    # Continue three-term recurrence
    axpy!(-alfa/beta, r2, r3)  # r3 = r3 - (alfa/beta) * r2
    copyto!(r1, r2)  # r1 = r2
    copyto!(r2, r3)  # r2 = r3
    
    # Compute next beta (with preconditioning if applicable)
    if Pl !== nothing
        ldiv!(r3, Pl, r2)  # r3 = M \ r2
        betan = sqrt(real(dot(r2, r3)))
        if betan < 0
            error("Preconditioner appears to be indefinite or singular")
        end
    else
        betan = norm(r2)
    end
    
    # Check for breakdown in first iteration
    if iteration == 1 && Pl === nothing
        if betan == 0
            if alfa == 0
                # A*b = 0 and x = 0
                phase = :stop
                ws_updated = update(ws; iteration, current=zero(real(eltype(x))))
                return x, ws_updated, phase
            else
                # A*b = alfa*b, x = b/alfa (eigenvector case)
                x .= b ./ alfa
                phase = :stop
                ws_updated = update(ws; iteration, current=zero(real(eltype(x))))
                return x, ws_updated, phase
            end
        end
    end
    
    # Apply previous left rotation Q_{k-1}
    dbar = state.dltan
    dlta = cs * dbar + sn * alfa
    epln = state.eplnn
    gbar = sn * dbar - cs * alfa
    eplnn = sn * betan
    dltan = -cs * betan
    dlta_qlp = dlta
    
    # Compute current left plane rotation Q_k
    gamal3 = state.gamal2
    gamal2 = state.gamal
    gamal = state.gama
    cs, sn, gama = sym_givens(gbar, betan)
    gama_tmp = gama
    
    taul2 = state.taul
    taul = tau
    tau = cs * phi
    phi = sn * phi
    
    # Apply previous right plane rotation P{k-2,k}
    if iteration > 2
        veplnl2 = state.veplnl
        etal2 = state.etal
        etal = state.eta
        dlta_tmp = state.sr2 * state.vepln - state.cr2 * dlta
        veplnl = state.cr2 * state.vepln + state.sr2 * dlta
        dlta = dlta_tmp
        eta = state.sr2 * gama
        gama = -state.cr2 * gama
    else
        veplnl2 = state.veplnl2
        etal2 = state.etal2
        etal = state.etal
        veplnl = state.veplnl
        eta = state.eta
    end
    
    # Compute current right plane rotation P{k-1,k}
    if iteration > 1
        cr1, sr1, gamal = sym_givens(gamal, dlta)
        vepln = sr1 * gama
        gama = -cr1 * gama
    else
        cr1 = state.cr1
        sr1 = state.sr1
        vepln = state.vepln
    end
    
    # Update xnorm
    xnorml = state.xnorm
    ul4 = state.ul3
    ul3 = state.ul2
    
    if iteration > 2
        ul2 = (taul2 - etal2 * ul4 - veplnl2 * ul3) / gamal2
    else
        ul2 = state.ul2
    end
    
    if iteration > 1
        ul = (taul - etal * ul3 - veplnl * ul2) / gamal
    else
        ul = state.ul
    end
    
    xnorm_tmp = norm([state.xl2norm, ul2, ul])
    
    if abs(gama) > eps(real(eltype(x))) && xnorm_tmp < maxxnorm
        u = (tau - eta * ul2 - vepln * ul) / gama
        if norm([xnorm_tmp, u]) > maxxnorm
            u = zero(real(eltype(x)))
            # Set flag for xnorm exceeded
        end    else
        u = zero(real(eltype(x)))
        # Set flag for singular system
    end
    
    xl2norm = norm([state.xl2norm, ul2])
    xnorm = norm([xl2norm, ul, u])
    
    # DEBUG: Print xnorm and maxxnorm values
    if iteration <= 10 || iteration % 10 == 0
        println("DEBUG MINRES iter $iteration: xnorm = $xnorm, maxxnorm = $maxxnorm, xnorm < maxxnorm: $(xnorm < maxxnorm)")
        println("  xl2norm = $xl2norm, ul = $ul, u = $u")
        println("  tau = $tau")
    end
    
    # Estimate condition number and norms
    pnorm = norm([betal, alfa, betan])
    abs_gama = abs(gama)
    anorml = state.anorm
    anorm = max(state.anorm, pnorm, gamal, abs_gama)
    
    if iteration == 1
        gmin = gama
    else
        gmin = min(state.gmin, gamal, abs_gama)
    end
    
    acond = anorm / gmin
    
    # Decide between MINRES and MINRES-QLP mode
    should_switch = (acond >= trancond) && minres_mode && (qlpiter == 0)
    
    if minres_mode && !should_switch
        # MINRES updates
        wl2_new = wl
        wl_new = w
        # w = (v - epln*wl2 - dlta_qlp*wl) / gama_tmp
        w_new = similar(w)
        w_new .= v
        axpy!(-epln, wl2, w_new)
        axpy!(-dlta_qlp, wl, w_new)
        w_new ./= gama_tmp
        
        # DEBUG: Additional debug info before solution update
        if iteration <= 10 || iteration % 10 == 0
            println("  About to update solution: tau = $tau, ||w_new|| = $(norm(w_new))")
        end
        
        if xnorm < maxxnorm
            println("  Before solution update: x[1:3] = ", x[1:3])
            axpy!(tau, w_new, x)  # x = x + tau * w
            println("  After solution update: x[1:3] = ", x[1:3])
            println("  tau * w_new[1:3] = ", tau .* w_new[1:3])
            if iteration <= 10 || iteration % 10 == 0
                println("  Solution updated! New ||x|| = $(norm(x))")
            end
        else
            if iteration <= 10 || iteration % 10 == 0
                println("  WARNING: Solution update SKIPPED due to xnorm >= maxxnorm")
            end
        end
        
        copyto!(wl2, wl2_new)
        copyto!(wl, wl_new)
        copyto!(w, w_new)
        
        new_qlpiter = qlpiter
        new_minres_mode = true
    else
        # MINRES-QLP updates
        new_qlpiter = qlpiter + 1
        new_minres_mode = false
        
        if new_qlpiter == 1
            # Transition to QLP mode - reconstruct direction vectors
            xl2 = zeros(eltype(x), length(x))
            if iteration > 1
                if iteration > 3
                    wl2_temp = gamal3 * wl2 + veplnl2 * wl + etal * w
                    copyto!(wl2, wl2_temp)
                end
                if iteration > 2
                    wl_temp = state.gamal_qlp * wl + state.vepln_qlp * w
                    copyto!(wl, wl_temp)
                end
                w_temp = state.gama_qlp * w
                copyto!(w, w_temp)
                xl2 .= x .- wl .* state.ul_qlp .- w .* state.u_qlp
            end
        else
            xl2 = x .- wl .* ul .- w .* u
        end
        
        # Update direction vectors for QLP
        if iteration == 1
            wl2_new = wl
            wl_new = v .* sr1
            w_new = .-v .* cr1
        elseif iteration == 2
            wl2_new = wl
            wl_new = w .* cr1 .+ v .* sr1
            w_new = w .* sr1 .- v .* cr1
        else
            wl2_new = wl
            wl_new = w
            w_new = wl2 .* state.sr2 .- v .* state.cr2
            wl2_temp = wl2 .* state.cr2 .+ v .* state.sr2
            v_temp = wl_new .* cr1 .+ w_new .* sr1
            w_new = wl_new .* sr1 .- w_new .* cr1
            wl_new = v_temp
            copyto!(wl2, wl2_temp)
        end
        
        # Update solution
        xl2 .+= wl2_new .* ul2
        x .= xl2 .+ wl_new .* ul .+ w_new .* u
        
        copyto!(wl2, wl2_new)
        copyto!(wl, wl_new)
        copyto!(w, w_new)
    end
    
    # Compute next right plane rotation P{k-1,k+1}
    gamal_tmp = gamal
    cr2, sr2, gamal = sym_givens(gamal, eplnn)
    
    # Store quantities for transferring from MINRES to MINRES-QLP
    gamal_qlp = gamal_tmp
    vepln_qlp = vepln
    gama_qlp = gama
    ul_qlp = ul
    u_qlp = u
      # Update residual norm estimate
    rnorm = phi
    current = rnorm
    # Check convergence using correct MATLAB formula
    relres = rnorm / (anorm * xnorm + state.beta1)
    converged = relres <= ws.options.reltol
    
    if converged || iteration >= ws.options.iterations
        phase = :stop
    else
        phase = :advance
    end
    
    # Update state
    ws_updated = update(ws;
        iteration, current,
        beta, betal, betan, alfa,
        cs, sn, cr1, sr1, cr2, sr2,
        dltan, eplnn, gama, gamal, gamal2,
        eta, etal, etal2, vepln, veplnl, veplnl2,
        ul, ul2, ul3, ul4, u, xnorm, xl2norm,
        anorm, acond, gmin, phi, tau, taul, taul2,
        qlpiter=new_qlpiter, minres_mode=new_minres_mode,
        gamal_qlp, vepln_qlp, gama_qlp, ul_qlp, u_qlp
    )
    
    print_progress(ws_updated)
    
    return x, ws_updated, phase
end
