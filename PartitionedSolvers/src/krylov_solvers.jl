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
struct MINRESState{A, B, C, D} <: AbstractType
    r::A  # Residual vector
    p0::A # Search direction (becomes p_k+1 after computation)
    p1::A # Search direction (becomes p_k after shift)
    p2::A # Search direction (becomes p_k-1 after shift)
    s0::A # A*p0
    s1::A # A*p1
    s2::A # A*p2
    A::B # System matrix
    iteration::C
    current::D # Current residual norm
    target::D  # Target residual norm
end

function minres_state(p)
    x_prototype = solution(p) # Used for type and size inference
    A_matrix = matrix(p)

    r  = similar(x_prototype, axes(A_matrix, 1))
    p0 = similar(x_prototype, axes(A_matrix, 2))
    p1 = similar(x_prototype, axes(A_matrix, 2))
    p2 = similar(x_prototype, axes(A_matrix, 2))
    s0 = similar(x_prototype, axes(A_matrix, 1))
    s1 = similar(x_prototype, axes(A_matrix, 1))
    s2 = similar(x_prototype, axes(A_matrix, 1))

    iteration = 0
    current_val = zero(real(eltype(x_prototype))) # Renamed to avoid conflict
    target_val  = zero(real(eltype(x_prototype))) # Renamed to avoid conflict

    MINRESState(r, p0, p1, p2, s0, s1, s2, A_matrix, iteration, current_val, target_val)
end

function update(o::MINRESState; kwargs...)
    data = (;kwargs...)
    r = hasproperty(data, :r) ? data.r : o.r
    p0 = hasproperty(data, :p0) ? data.p0 : o.p0
    p1 = hasproperty(data, :p1) ? data.p1 : o.p1
    p2 = hasproperty(data, :p2) ? data.p2 : o.p2
    s0 = hasproperty(data, :s0) ? data.s0 : o.s0
    s1 = hasproperty(data, :s1) ? data.s1 : o.s1
    s2 = hasproperty(data, :s2) ? data.s2 : o.s2
    A_mat = hasproperty(data, :A) ? data.A : o.A # Renamed to avoid conflict
    iter = hasproperty(data, :iteration) ? data.iteration : o.iteration # Renamed
    curr = hasproperty(data, :current) ? data.current : o.current # Renamed
    targ = hasproperty(data, :target) ? data.target : o.target # Renamed
    MINRESState(r, p0, p1, p2, s0, s1, s2, A_mat, iter, curr, targ)
end

function minres(p; kwargs...)
    options = krylov_options(p; kwargs...)
    state = minres_state(p)
    workspace = KrylovWorkspace(options, state)
    linear_solver(minres_update, minres_step, p, workspace)
end

function minres_update(ws, current_A_matrix)
    options = ws.options
    Pl_updated = options.Pl # Get current Pl

    if options.update_Pl && options.Pl !== nothing
        # This assumes Pl has an update method like: new_Pl = update(old_Pl; matrix=current_A_matrix)
        # This part is highly dependent on the specific preconditioner's API.
        Pl_updated = update(options.Pl; matrix=current_A_matrix)
    end

    iteration = 0
    # Update workspace options if Pl changed, and state for A and iteration.
    ws = update(ws; iteration, A=current_A_matrix, Pl=Pl_updated)
    return ws
end

function minres_step(x, ws, b, phase=:start; kwargs...)
    # Extract from workspace state (MINRESState)
    # Note: A is part of ws.state.A
    minres_st = ws.state # Renamed to avoid conflict with A field
    A_mat = minres_st.A    # System matrix from state
    r, p0, p1, p2 = minres_st.r, minres_st.p0, minres_st.p1, minres_st.p2
    s0, s1, s2 = minres_st.s0, minres_st.s1, minres_st.s2
    iteration = minres_st.iteration
    # current, target are updated and passed back via update(ws; ...)

    # Extract from workspace options
    opt = ws.options # Renamed to avoid conflict
    reltol, abstol, norm_func, Pl = opt.reltol, opt.abstol, opt.norm, opt.Pl # norm_func

    if phase === :start
        iteration = 0
        phase = :advance

        mul!(s2, A_mat, x)      # s2 = A*x (use s2 as temp)
        copyto!(r, b)       # r = b
        axpy!(-one(eltype(s2)), s2, r) # r = r - s2 = b - A*x

        current_norm = norm_func(r) # Use norm_func
        target_norm = max(reltol * current_norm, abstol)

        copyto!(p1, r)
        mul!(s1, A_mat, p1)
        copyto!(p0, p1)
        copyto!(s0, s1)

        fill!(p2, zero(eltype(p2)))
        fill!(s2, zero(eltype(s2)))

        ws = update(ws; iteration, current=current_norm, target=target_norm, r, p0, p1, p2, s0, s1, s2, A=A_mat)
        print_progress_header(ws)
    end

    copyto!(p2, p1)
    copyto!(s2, s1)
    copyto!(p1, p0)
    copyto!(s1, s0)

    alpha_num = dot(r, s1)
    alpha_den = dot(s1, s1)
    α = ifelse(iszero(alpha_den), zero(eltype(r)), alpha_num / alpha_den)

    axpy!(α, p1, x)
    axpy!(-α, s1, r)

    copyto!(p0, s1)
    mul!(s0, A_mat, p0)

    # Orthogonalization coefficients (β1, β2)
    # Note: The denominators for β1 and β2 are based on the original MINRES formulation.
    # alpha_den (dot(s1,s1)) is ||A*p1_current||^2 if p1 is normalized, or (A*p1)^T (A*p1)
    # This was used for β1 in the original code.
    #den_s1_s1 = dot(s1,s1) 
    #beta1_num = dot(s0, s1) # s0 is A*(A*p1_current), s1 is A*p1_current
    #β1 = ifelse(iszero(den_s1_s1), zero(eltype(r)), beta1_num / den_s1_s1)

    beta1_num = dot(s0, s1)
    β1 = ifelse(iszero(alpha_den), zero(eltype(r)), beta1_num / alpha_den) # Reuse alpha_den

    axpy!(-β1, p1, p0)
    axpy!(-β1, s1, s0)

    if iteration > 0
        beta2_num = dot(s0, s2)
        beta2_den = dot(s2, s2)
        β2 = ifelse(iszero(beta2_den), zero(eltype(r)), beta2_num / beta2_den)

        axpy!(-β2, p2, p0)
        axpy!(-β2, s2, s0)
    end
    
    # If ldiv!(p0, Pl, p0) is not safe for the specific Pl (e.g., iterative solver Pl),
    # a temporary vector would be needed: temp = copy(p0); ldiv!(p0, Pl, temp).
    # For now, assume ldiv!(p0, Pl, p0) is acceptable.
    if Pl !== nothing && Pl !== identity # Add check if Pl is effective
        ldiv!(p0, Pl, p0) # p0 = Pl \ p0 (where p0 was q_k)
        mul!(s0, A_mat, p0) # Recompute s0 = A * (new preconditioned p0)
    end

    current_norm = norm_func(r) # Use norm_func
    iteration += 1

    ws = update(ws; iteration, current=current_norm, r, p0, p1, p2, s0, s1, s2) # target is not changed here
    print_progress(ws)

    if done(ws)
        phase = :stop
    end

    return x, ws, phase
end

function minres_solve!(A, b, x0, tol)
    x  = copy(x0)
    r  = b .- A * x
    p0 = copy(r)
    s0 = A * p0
    p1 = copy(p0)
    s1 = copy(s0)

    for iter in 1:1000
        # shift the p‐ and s‐sequences
        p2, p1 = p1, p0
        s2, s1 = s1, s0

        # step length
        α = dot(r, s1) / dot(s1, s1)

        # update solution and residual
        @. x += α * p1
        @. r -= α * s1

        # check convergence
        if dot(r, r) < tol^2
            break
        end

        # build new p0, s0
        p0 = copy(s1)
        s0 = A * s1

        β1 = dot(s0, s1) / dot(s1, s1)
        @. p0 -= β1 * p1
        @. s0 -= β1 * s1

        if iter > 1
            β2 = dot(s0, s2) / dot(s2, s2)
            @. p0 -= β2 * p2
            @. s0 -= β2 * s2
        end
        
        #if iter == 2
        #    println("--- minres_solve! after 1st iteration ---")
        #    println("x: ", x)
        #    println("r: ", r)
        #    println("p0 (new p_k+1): ", p0)
        #    println("p1 (p_k): ", p1)
        #    println("p2 (p_k-1, from start of iter): ", iter > 1 ? p2 : "N/A or initial p2")
        #    println("s0 (new A*p_k+1): ", s0)
        #    println("s1 (A*p_k): ", s1)
        #    println("s2 (A*p_k-1, from start of iter): ", iter > 1 ? s2 : "N/A or initial s2")
        #    println("α: ", α)
        #    println("β1: ", β1)
        #    println("β2: ", iter > 1 ? β2 : "N/A (not computed in 1st iter)")
        #    println("--------------------------------------")
        #end

    end

    return x, r
end


# 1. Compare the wiki solver first converge step with the AI developed solver, make sure they are the same. Make sure there are no allocations.
# 2. Implement simple tests with PETSc and this solver if 1. is correct
# 3. Find matricies for analasis, and implement them to analyse the solver and compare with PETSc
# 4. Prepare the files to be used in DAS-5
# 5. Gather data from DAS-5 runs