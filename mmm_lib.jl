# Version 0.1, 2022
# (c) Rahul Siddharthan, 2023
# Run with -h for options
# Licence: MIT

# struct containing global params

mutable struct Params
    minclustsize::Int64
    c::Float64
    µ₀::Float64
    β₀::Float64
    a₀::Float64
    b₀::Float64
    rows_c::Array{Int64,2} # this will store the raw *categorical* rows being clustered
    rows_n::Array{Float64,2} # this will store the raw *normal* rows being clustered
    nrows::Int64 # number of rows, in total
    rowcluster::Vector{Int64} # says which cluster n'th row belongs to
                              # 0 = not clustered (temporarily popped)
    datavec_c::Vector{Int64}
    datavec_n::Vector{Float64}
    typevec::Vector{Int64} # only for categorical, = k = #categories
    # used by clusterLikelihood, avoid reallocating 
    lrow_c::Int64 # length of categorical row
    lrow_n::Int64 # length of normal row
    maxnclust::Int64
    nclust::Int64 # if you want to specify exactly how many clusters needed
    quiet::Bool
    debug::Bool
    beta::Float64 # in sampleClusters to reduce bias, beta < 1
    niter::Int64 # in sampleClusters
end


# row struct no longer needed?        
#mutable struct  Row # a row of heterogeneous data
#    rowindex::Int64 # the line corresponding to the row data
#    clustid::Int64 #used in output
#end

mutable struct Cluster
    rows::Vector{Bool} # if rows[n] then n is in this cluster
    nrows::Int64
    clustid::Int64 # unique ID for each cluster
    xm::Vector{Float64} # to calculate normal-gamma parameters, pre-stored
    Σx²::Vector{Float64} # to calculate normal-gamma parameters, pre-stored    
    μ::Vector{Float64} # normal-gamma parameters, vector of length glob.typevec, 
    β::Vector{Float64}
    a::Vector{Float64}
    b::Vector{Float64}
    Λ::Vector{Float64} # = (a*β)/b/(β+1)
    Λfac::Vector{Float64} # = π^(-0.5)*gammarat*(Λ/2/a)^0.5
    counts::Array{Int64,2} # dirichlet counts, height k = maximum in typevec,
                           # length = glob.lrow_c
end

function posteriorParamsNG(X::SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true},glob::Params)
    # X = vector of observations 
    # given prior parameters and a vector of data,
    # this calculates posterior parameters 
    # to be used in calculating the likelihood
    μ₀, β₀, a₀, b₀ = glob.μ₀, glob.β₀, glob.a₀, glob.b₀ 
    if length(X)==0
        return μ₀, β₀, a₀, b₀ 
    else
        n = length(X)
        xm = sum(X)/n
        µn = (β₀*μ₀ + n*xm)/(β₀+n)
        βn = β₀ + n
        an = a₀ + n/2
        bn = b₀ + sum([(X[i]-xm)^2 for i in 1:n])/2 + (β₀*n*(xm-µ₀)^2)/2/(β₀+n)
        return µn, βn, an, bn
    end
end

function log_posterior_predictive_NG(x::Float64,µ::Float64,β::Float64,
                                 a::Float64,b::Float64,Λ::Float64,
                                 Λfac::Float64) # the parameters need to be calculated using posteriorParamsNG 
    # Λ = (a*β)/b/(β+1)
    # gammarat = exp(loggamma(a+0.5)-loggamma(a))
    # Λfac = log(π^(-0.5)*gammarat*(Λ/2/cl.a[i])^0.5)
    return Λfac- (a+0.5)*log(1+(Λ*(x-µ)^2)/2/a)
end


function log_marginal_likelihood_NG(cl::Cluster,i::Int64,glob::Params)
    μ₀ = glob.µ₀
    β₀ = glob.β₀
    a₀ = glob.a₀
    b₀ = glob.b₀
    n = cl.nrows
    µn = cl.μ[i]
    βn = cl.β[i]
    an = cl.a[i]
    bn = cl.b[i]
    gammarat = loggamma(an)-loggamma(a₀)
    return gammarat + a₀*log(b₀) - an*log(bn) + 0.5*log(β₀/βn) -n/2*log(2*π)
end



function log_marginal_likelihood_dirichlet(cl::Cluster, i::Int64, glob::Params)
    counts = @view cl.counts[:,i]
    k = glob.typevec[i]
    c = glob.c
    ll = lgamma(4*c) - 4*lgamma(c) - lgamma(cl.nrows+4*c)
    for m in 1:k
        ll += lgamma(counts[m]+c)
    end
    return ll
end


#read tabular data
function parse_csv_line_floats(l)
    return [parse(Float64,x) for x in split(l,",")]
end

function parseIorF(x)
    if occursin(".",x)
        return parse(Float64,x)
    else
        return parse(Int64,x)
    end
end

function parse_csv_line_mixed(l)
    tokens = split(l,",")
    parsed_c = Int64[]
    parsed_n = Float64[]
    for t in tokens
        if occursin(".",t)
            push!(parsed_n,parse(Float64,t))
        else
            push!(parsed_c,parse(Int64,t))
        end
    end
    return parsed_c, parsed_n
end

function get_typevec(rows_c) # only for categorical
    typevec::Vector{Int64} = Int64[]
    for n = 1:length(@view rows_c[:,1])
        push!(typevec,maximum(@view rows_c[n,:]))
    end
    return typevec
end


    
# return a cluster with no sequences in it
# clustid = 0 for null cluster, set it to something else for actual cluster
# FIXME
function setupNullCluster(glob::Params)
    xmvec = [0.0 for i in 1:glob.lrow_n]
    Σx²vec = [0.0 for i in 1:glob.lrow_n]    
    μvec = [glob.μ₀ for i in 1:glob.lrow_n]
    βvec = [glob.β₀ for i in 1:glob.lrow_n]
    avec = [glob.a₀ for i in 1:glob.lrow_n]
    bvec = [glob.b₀ for i in 1:glob.lrow_n]
    Λ = (glob.a₀*glob.β₀)/glob.b₀/(glob.β₀+1)
    gammarat = exp(loggamma(glob.a₀+0.5)-loggamma(glob.a₀))
    Λfac = π^(-0.5)*gammarat*(Λ/2/glob.a₀)^0.5    
    Λvec = [Λ for i in 1:glob.lrow_n]
    Λfacvec = [log(Λfac) for i in 1:glob.lrow_n]    
    if glob.lrow_c > 0
        cvec = zeros(Int64,maximum(glob.typevec),glob.lrow_c)
    else
        cvec = zeros(Int64,1,1)
    end
    return Cluster([false for i=1:glob.nrows],0, 0, xmvec,Σx²vec,μvec,βvec,avec,bvec,Λvec,Λfacvec,cvec)
end

# given an array of sequences, initialize parameters with seq cluster in one seq
function initParams!(rowlist_n::Array{Float64,2},rowlist_c::Array{Int64,2},glob::Params)
    glob.typevec = get_typevec(rowlist_c)
    glob.lrow_c = length(glob.typevec)
    glob.lrow_n = length(@view rowlist_n[:,1])
    if glob.lrow_c>0
        glob.nrows = length(@view rowlist_c[1,:])
    else
        glob.nrows = length(@view rowlist_n[1,:])
    end
    glob.datavec_c = [0 for i in 1:glob.nrows]
    glob.datavec_n = [0.0 for i in 1:glob.nrows]    
    glob.rows_c = rowlist_c
    glob.rows_n = rowlist_n
    cl = setupNullCluster(glob)
    cl.clustid = 1
    cl.nrows = 0
    glob.rowcluster = [0 for i in 1:glob.nrows]    
    for n in 1:glob.nrows
        pushCluster!(cl,n,glob)
    end
    return cl
end

# float or int
is_a_num(x) = isa(tryparse(Float64,x),Number) 

# float not int
is_a_float(x) = isa(tryparse(Float64,x),Number) && !isa(tryparse(Int64,x),Number)

function indexin_str(token::SubString{String},stringlist::Vector{SubString{String}})
    for n = 1:length(stringlist)
        if token==stringlist[n]
            return n
        end
    end
    return 0
end
    

function read_csv(filename::String,typestring::String,forcefloat::Bool,excludelast::Bool)
    # make this function flexible
    # lines beginning with # are ignored
    # if filename ends with .csv it's a csv file
    # if file ends with .tsv it's a tsv file
    # read whole file into a matrix of strings
    # then infer which columns are real and which are categorical
    # categorical need not be 1,2,3...
    # but return a table where categorical is 1,2,3... and inferred typevec
    # and also a map from the numerical categories to the original categories
    f = open(filename,"r")
    if filename[end-2:end]=="csv"
        token = ","
    elseif filename[end-2:end]=="tsv"
        token = "\t"
    else # default = tsv
        token = "XXX" # placeholder, figure it out below
    end
    linesraw = [l for l in readlines(f) if l[1]!='#']
    close(f)
    if '\t' in linesraw[1]
        token = "\t"
    elseif ',' in linesraw[1]
        token = ","
    elseif ' ' in linesraw[1]
        token = " "
    else
        error("Couldn't figure out separator in "*filename*": ensure it is CSV, TSV or space separated")
    end
    flines = [split(l,token) for l in linesraw]
        
    lrow = length(flines[1])
    if any(map(x->length(x)!=lrow,flines))
        error("Inconsistent line lengths in file $filename")
    end
    if excludelast
        flines = [l[1:end-1] for l in flines]
        lrow -= 1
    end
    nrows = length(flines)
    rowsf::Vector{Vector{Float64}} = [Float64[] for n in 1:nrows]
    rowsc::Vector{Vector{Int64}} = [Int64[] for n in 1:nrows]
    for i in 1:lrow        
        tokens = [l[i] for l in flines]
        catvars = unique(tokens)
        if forcefloat
            isf = true
        elseif length(typestring) > 0
            if typestring[i] == 'N' || typestring[i] == 'n'
                isf = true
            elseif typestring[i] == 'C' || typestring[i] == 'c'
                isf = false
            else
                throw("Invalid typestring")
            end
        else
            isf = any(map(x->is_a_float(x),tokens)) && all(map(x->is_a_num(x),tokens))
            if ~isf && all(map(x->is_a_num(x),tokens)) && length(catvars) >= 10
                # assume it is a float, because there are too many categories:arbitrary! FIXME
                isf = true
            end
        end
        if isf
            # real if all numbers, at least one has a decimal point
            for j in 1:nrows
                push!(rowsf[j],parse(Float64,tokens[j]))
            end
        else # categorical, count categories
            for j in 1:nrows
                push!(rowsc[j],Int64(indexin_str(tokens[j],catvars)[1]))
            end
        end
    end
    return Matrix(reduce(hcat,rowsf)),Matrix(reduce(hcat,rowsc))
end

function init_default(filename,typestring,forcefloat,excludelast)
    glob = Params(0,# minclustsize
                  0.5, # c
                  0.0, # μ₀
                  0.5, # β₀
                  0.5, # a₀
                  0.5, # b₀
                  zeros(Int64,2,2), # rows_c
                  zeros(Float64,2,2), # rows_n
                  0,  # nrows
                  [0], # rowcluster
                  [0], # datavec_c
                  [0], #datavec_n
                  [0], #typevec
                  0, #lrow_c
                  0, #lrow_n
                  0, #maxnclust
                  0, #nclust
                  true, #quiet
                  false, #debug
                  0.5, #beta
                  10 #niter
                  )
    rowsf,rowsc = read_csv(filename,typestring,forcefloat,excludelast)
    cl = initParams!(rowsf, rowsc, glob)
    return glob, cl
end


function init_default_old(filename)
    glob = Params(0,# minclustsize
                  0.5, # c
                  0.0, # μ₀
                  0.5, # β₀
                  0.5, # a₀
                  0.5, # b₀
                  zeros(Int64,2,2), # rows_c
                  zeros(Float64,2,2), # rows_n
                  0,  # nrows
                  [0], # rowcluster
                  [0], # datavec_c
                  [0], #datavec_n
                  [0], #typevec
                  0, #lrow_c
                  0, #lrow_n
                  0, #maxnclust
                  0, #nclust
                  true, #quiet
                  false, #debug
                  0.5, #beta
                  10 #niter
                  )
                   
    f = open(filename,"r")
    rows_c1 = Int64[]
    rows_n1 = Float64[]
    lc = 0
    ln = 0
    for l in readlines(f)
        row_c, row_n = parse_csv_line_mixed(l)
        lc = length(row_c)
        ln = length(row_n)
        append!(rows_c1, row_c)
        append!(rows_n1, row_n)
    end
    close(f)
    if lc>0
        nlinesc = length(rows_c1) ÷ lc
    else
        nlinesc = 0
    end
    if ln>0
        nlinesn = length(rows_n1) ÷ ln
    else
        nlinesn = 0
    end
    if nlinesc==0 && nlinesn==0
        error("init_default: zero data?\n")
    elseif nlinesc==0
        nlinesc=nlinesn
    elseif nlinesn==0
        nlinesn=nlinesc
    end
    if nlinesc != nlinesn
        error("init_default: inconsistent # lines for categ (",nlinesc,") and normal (",nlinesn,")\n")
    end
    rows_c = reshape(rows_c1,lc,nlinesc)
    rows_n = reshape(rows_n1,ln,nlinesn)
    cl = initParams!(rows_n, rows_c, glob)
    return glob, cl
end



function updateClusterParams_full!(cl::Cluster,glob::Params)
    μ₀, β₀, a₀, b₀ = glob.μ₀, glob.β₀, glob.a₀, glob.b₀ 
    for i in 1:glob.lrow_n
        if cl.nrows==0
            cl.xm[i] = 0.0
            cl.Σx²[i] = 0.0
            cl.μ[i] = μ₀
            cl.β[i] = β₀
            cl.a[i] = a₀
            cl.b[i] = b₀ 
        else #FIXME DOUBLECHECK THIS
            n = cl.nrows
            xm = sum([glob.rows_n[i,j] for j in 1:glob.nrows if cl.rows[j]])/n
            Σx² = sum([glob.rows_n[i,j]^2 for j in 1:glob.nrows if cl.rows[j]])
            cl.μ[i] = (β₀*μ₀ + n*xm)/(β₀+n)
            cl.β[i] = β₀ + n
            cl.a[i] = a₀ + n/2
            cl.b[i] = b₀ + 0.5*(Σx² - n*xm^2) + (β₀*n*(xm-µ₀)^2)/2/(β₀+n)
            cl.xm[i] = xm
            cl.Σx²[i] = Σx²
        end
        gammarat = exp(loggamma(cl.a[i]+0.5)-loggamma(cl.a[i]))
        cl.Λ[i] =  (cl.a[i]*cl.β[i])/cl.b[i]/(cl.β[i]+1)
        cl.Λfac[i] = log(π^(-0.5)*gammarat*(cl.Λ[i]/2/cl.a[i])^0.5)
    end
    for i in 1:glob.lrow_c
        for k=1:glob.typevec[i]
            cl.counts[k,i] = 0
        end
        for j in 1:glob.nrows
            if cl.rows[j]
                cl.counts[glob.rows_c[i,j],i] += 1
            end
        end
    end
end



# note change: this is called BEFORE adding/removing row to cluster
# pop==true means removing row, pop==false means adding row
function updateClusterParams!(cl::Cluster,glob::Params,nrow::Int64,ispop::Bool)
    μ₀, β₀, a₀, b₀ = glob.μ₀, glob.β₀, glob.a₀, glob.b₀
    x = 0.0
    xc = 0
    xm = 0.0
    Σx² = 0.0
    for i in 1:glob.lrow_n
        x = glob.rows_n[i,nrow]
        if cl.nrows==1 && ispop # this means the cluster will become empty
            cl.xm[i] = 0.0
            cl.Σx²[i] = 0.0
            cl.μ[i] = μ₀
            cl.β[i] = β₀
            cl.a[i] = a₀
            cl.b[i] = b₀ 
        else #FIXME DOUBLECHECK THIS
            n = cl.nrows 
            if ispop # note: nrows is really n-1, not yet updated
                xm = (n*cl.xm[i] - x)/(n-1)
                Σx² =  cl.Σx²[i] - x^2
                # note: row has already been popped from cl.rows
                cl.μ[i] = (β₀*μ₀ + (n-1)*xm)/(β₀+n-1)
                cl.β[i] -= 1
                cl.a[i] -= 0.5
                cl.b[i] = b₀ + 0.5*(Σx²-(n-1)*xm^2) + (β₀*(n-1)*(xm-µ₀)^2)/2/(β₀+n-1)
                cl.xm[i] = xm
                cl.Σx²[i] = Σx²
            elseif cl.nrows==0 # and ispop==false
                xm = x
                Σx² = x^2
                cl.μ[i] = (β₀*μ₀ + xm)/(β₀+1)
                cl.β[i] += 1
                cl.a[i] += 0.5
                cl.b[i] = b₀ + (β₀*(xm-µ₀)^2)/2/(β₀+1)
                cl.xm[i] = xm
                cl.Σx²[i] = Σx²
            else
                xm = (n*cl.xm[i] + x)/(n+1)                    
                Σx² = cl.Σx²[i] + x^2
                
                cl.μ[i] = (β₀*μ₀ + (n+1)*xm)/(β₀+n+1)
                cl.β[i] += 1
                cl.a[i] += 0.5
                cl.b[i] = b₀ + 0.5*(Σx²-(n+1)*xm^2) + (β₀*(n+1)*(xm-µ₀)^2)/2/(β₀+n+1)
                cl.xm[i] = xm
                cl.Σx²[i] = Σx²                
            end
        end
        gammarat = exp(loggamma(cl.a[i]+0.5)-loggamma(cl.a[i]))
        cl.Λ[i] =  (cl.a[i]*cl.β[i])/cl.b[i]/(cl.β[i]+1)
        cl.Λfac[i] = log(π^(-0.5)*gammarat*(cl.Λ[i]/2/cl.a[i])^0.5)
    end
    for i = 1:glob.lrow_c
        xc = glob.rows_c[i,nrow]
        if ispop
            cl.counts[xc,i] -= 1
        else
            cl.counts[xc,i] += 1
        end
    end
end


# add a sequence to a cluster
function pushCluster!(cl::Cluster, nrow::Int64,glob::Params)
    # TODO can hyperparameters c, µn, etc, be kept here and updated
    # efficiently?
    updateClusterParams!(cl,glob,nrow,false)
    cl.rows[nrow] = true
    cl.nrows += 1
    glob.rowcluster[nrow] = cl.clustid
    #= if cl.nrows != length([x for x in cl.rows if x])
        error("pushCluster: nrows=",cl.nrows," but ", length([x for x in cl.rows if x])," true")
    end
    =#
end

# remove sequence with id nrow from the cluster
function spliceCluster!(cl::Cluster,nrow::Int64,glob::Params)
    cl.rows[nrow] = false
    updateClusterParams!(cl,glob,nrow,true)
    cl.nrows -= 1
    glob.rowcluster[nrow] = 0
    #=
    if cl.nrows != length([x for x in cl.rows if x])
        error("popCluster: nrows=",cl.nrows," but ", length([x for x in cl.rows if x])," true")
    end
    =#
end




# given an array of sequences, set up a seq cluster
#=
function setupCluster(rowlist::Array{Float64,2},glob::Params,clustid::Int64)
    cl = setupNullCluster(glob)
    cl.clustid=clustid
    for n in 1:length(rowlist)
        r = rowlist[n]
        row1 = Row(r,n,1,0)
        row1.clustid=clustid
        pushCluster!(cl,row1,glob)
    end
    return cl
end
=#

function likelihood_Dirichlet(i::Int64,counts::Array{Int64,2},j::Int64,glob::Params)
    N = @view counts[:,j]
    return (N[i]+glob.c)/(sum(N)+length(N)*glob.c)
end

function rowLikelihood(row::Int64, cl::Cluster, glob::Params)
    # typevec = same length as datarow, 0 = continuous, > 0 = categorical k-valued 
    # c = pseudocount for categorical (all categorical = same pseudocount)
    # µ0 etc = parameters for normal-gamma prior (same for all real variables)
    # returns log likelihood, not likelihood!
    c = glob.c
    lik = 0.0
    for i in 1:glob.lrow_n
        lik += log_posterior_predictive_NG(glob.rows_n[i,row], cl.µ[i], cl.β[i], cl.a[i], cl.b[i],cl.Λ[i],cl.Λfac[i])
    end
    for i in 1:glob.lrow_c
        lik += log(likelihood_Dirichlet(glob.rows_c[i,row],cl.counts,i,glob))
    end
    if isinf(lik) || isnan(lik)
        println("rowLikelihood: ",lik)
        println(row," ",glob.rows_c[:,row]," ",glob.rows_n[:,row])
        println(cl)
        error("rowLikelihood: problem")
    end
    return lik
end



# log likelihood ratio of the sequences in cl1 (and cl2 if
# supplied) all being sampled from the *same* distributions
function clusterLikelihood(glob::Params,cl::Cluster)
    ll = 0.0
    for n = 1:glob.lrow_n
        ll += log_marginal_likelihood_NG(cl,n,glob)
    end
    for n = 1:glob.lrow_c
        ll += log_marginal_likelihood_dirichlet(cl,n,glob)
    end
    return ll
end


# given a list of log likelihoods, return the average of log likelihoods
# this is log(avg(exp.(l))) but pull out a factor exp(m) first where m is
# largest value
function avg_logliklist(l::Vector{Float64})
    m = maximum(l)
    return m + log(sum([exp(x-m) for x in l])/length(l))
end


function pickFromScores(scores::Vector{Float64})
    # this is a log likelihood list.
    m = maximum(scores)
    scores1 = [exp(x-m) for x in scores]
    scores1 /= sum(scores1)
    r = rand()
    n = 1
    tot = scores1[1]
    while r > tot && n<length(scores)
        n += 1
        tot += scores1[n]
    end
    return n
end



function save_clusters(glob::Params, clusters::Vector{Cluster})
    return [deepcopy(c.rows) for c in clusters]
end

function restore_clusters!(glob::Params, clusters::Vector{Cluster}, clusterids::Vector{Vector{Bool}})
    nclusters = length(clusterids)
    if nclusters > length(clusters)
        error("restore_clusters: max clusterids=",nclusters,"  but #clusters=",length(clusters))
    end
    while nclusters < length(clusters)
        pop!(clusters)
    end
    for n in 1:glob.nrows
        glob.rowcluster[n] = 0
    end
    for n in 1:nclusters
        for j in 1:glob.nrows
            clusters[n].rows[j] = clusterids[n][j]
            if clusterids[n][j]
                glob.rowcluster[j] = n
            end
        end
        clusters[n].clustid=n
        clusters[n].nrows = length([x for x in clusters[n].rows if x])
        updateClusterParams_full!(clusters[n],glob)
    end
end
                

## optimizeClusters_EM outline

## Latent variables Z_i with values 1..K, indicating cluster
## assignment of i. 

## In GMM, P(X_i = x) = \sum_k π_k P(X_i=x|Z_i=k)

## where π_k ≡ P(Z_i=k)
## note, π_k doesn't depend on i, but the posterior
## P(Z_i=k|X_i) ≡ γ_{Z_i} (k) =  P(x|C_k)π_k/Σ()
## does depend on i. 

## Maintain a vector π_{ik} giving probability of membership for each
## sequence in each cluster, initially uniform

## 1. Recalculate π using the current clustering
## 2. Assign each sequence to its best cluster using π
## Iterate until no reassignments.

## Randomly initialize split (or initialize from previous split)
## this function receives the random split


## thought: instead of assigning each row to its best cluster at each step,
## generate an ensemble of possible clusters. Pick one of those based
## on its clusterLikelihood, but pick from the likelihood distribution,
## not the max likelihood; with a fictitious temperature β that is increased
## slowly (simulated annealing). 


function optimizeClusters_EM!(glob::Params, clusters::Vector{Cluster})
    # sanity check on clusters
    nc = length(clusters)
    for c = 1:nc
        if clusters[c].clustid != c
            error("optimizeClusters_EM ERROR cluster IDs don't match: cluster ",c," has id ",clusters[c].clustid,"\n")
        end
        for n in 1:glob.nrows
            if clusters[c].rows[n] && glob.rowcluster[n] != c
                error("optimizeClustes_EM ERROR row ",n,"'s row index doesn't match its cluster ",c,"\n")
            end
        end
    end
    
    reassigned = true
    # nrows shoudl be total number of rows
    # nrows = sum([c.nrows for c in clusters])
    nrows = glob.nrows
    
    # π[i][j] = posterior prob. that i'th seq is in j'th cluster
    # initially uniform
    π = 1.0/nc .+ zeros(Float64,nc,nrows)
    π1 = 1.0/nc .+ zeros(Float64,nc,nrows)    
    while reassigned
        π1 .= 1.0/nc
        for n = 1:nrows
            cl = clusters[glob.rowcluster[n]]
            if cl.nrows < 100
                spliceCluster!(cl,n,glob)
            end
            for c2 = 1:nc
                π1[c2,n] = log(π[c2,n])+rowLikelihood(n,clusters[c2],glob)
            end
            if cl.nrows < 100
                pushCluster!(cl,n,glob)
            end
        end
        # was in log space, now exponentiate and normalize
        for n = 1:nrows
            maxπ1 = maximum(@view π1[:,n])
            for np = 1:nc
                π1[np,n] = exp(π1[np,n]-maxπ1)
            end
            #π1[n] = exp.(π1[n].-maxπ1) # allocs, avoid
            # is there a faster way to do this without allocation?
            πs = 0.0
            for x = 1:nc
                πs += π1[x,n]
            end
            for x = 1:nc
                π1[x,n] /= πs
            end
        end
        # assign each sequence to its most probable cluster
        reassigned = false
        for n = 1:glob.nrows
            c_old = glob.rowcluster[n]
            pmax = maximum(@view π1[:,n])
            newc = findall(x -> x==pmax,@view π1[:,n])
            if length(newc)==0
                println("optimizeClusters_EM: ",π1, " ", n)
                println("optimizeClusters_EM: ", pmax, " ", π1[:,n])
                error("optimizeClusters_EM: no clusters?")
            end
            if ~ (c_old in newc) # do nothing if c is there
                # pick random if more than one
                #shuffle!(newc)
                #c_new = newc[1]
                c_new = rand(newc)
                spliceCluster!(clusters[c_old],n,glob)
                pushCluster!(clusters[c_new],n,glob)
                reassigned = true
            end
        end
        # avoid reallocating
        π .= π1
    end
end


function get_bic(glob::Params,ll::Float64,clusters::Vector{Cluster})
    nparams = 0
    for c in glob.typevec
        nparams += c-1
    end
    nparams += 2*glob.lrow_n
    nparams *= length(clusters)
    # scikit-learn's nparams is adjusted by nclusters-1 : why? But try it
    nparams += length(clusters)-1
    return -2*ll +nparams*log(glob.nrows)
end

function getNclusters!(clusters::Vector{Cluster},glob::Params)
    # assume only one cluster in cluster, split it randomly
    for n = 1:glob.nclust-1
        newclust = setupNullCluster(glob)
        newclust.clustid = length(clusters)+1
        for m = 1:glob.nrows ÷ glob.nclust
            r = rand(1:glob.nrows)
            while clusters[1].rows[r]==false
                r = rand(1:glob.nrows)
            end
            spliceCluster!(clusters[1],r,glob)
            pushCluster!(newclust,r,glob)
        end
        println(newclust.nrows)
        push!(clusters,newclust)
    end
end


function addNewCluster!(clusters::Vector{Cluster},glob::Params)
    # add a new cluster from worst sequences of other clusters, prior to EM
    newclust = setupNullCluster(glob)
    newclust.clustid = length(clusters)+1
    # find 1/(N+1) worst sequences in existing clusters
    seqscorelist = Tuple{Float64,Int64}[]
    for n in 1:glob.nrows
        c = glob.rowcluster[n]
        if length(clusters) < c
            println(c, " ", length(clusters))
            println(glob.rowcluster)
            error("addNewCluster: rowcluster error")
        end
        spliceCluster!(clusters[c],n,glob)
        push!(seqscorelist,(rowLikelihood(n,clusters[c],glob),n))
        pushCluster!(clusters[c],n,glob)
    end
    nbadids = convert(Int64,round(length(seqscorelist)/newclust.clustid*0.9))
    #badids = [n for (c,n) in seqscorelist[1:length(seqscorelist)÷newclust.clustid]]
    badids = [n for (c,n) in seqscorelist[1:nbadids]]
    for n in 1:glob.nrows
        if n in badids
            spliceCluster!(clusters[glob.rowcluster[n]],n,glob)
            pushCluster!(newclust,n,glob)
        end
    end
    push!(clusters,newclust)
end

function addNewCluster_one!(clusters::Vector{Cluster},glob::Params, c::Int64)
    # add a new cluster from worst sequences of other clusters, prior to EM
    newclust = setupNullCluster(glob)
    newclust.clustid = length(clusters)+1
    # find 1/(N+1) worst sequences in existing clusters
    seqscorelist = Tuple{Float64,Int64}[]
    for n in 1:glob.nrows
        if glob.rowcluster[n]==c
            spliceCluster!(clusters[c],n,glob)
            push!(seqscorelist,(rowLikelihood(n,clusters[c],glob),n))
            pushCluster!(clusters[c],n,glob)
        end
    end
    nbadids = convert(Int64,round(length(seqscorelist)*0.3))
    #badids = [n for (c,n) in seqscorelist[1:length(seqscorelist)÷newclust.clustid]]
    badids = [n for (c,n) in seqscorelist[1:nbadids]]
    for n in 1:glob.nrows
        if n in badids
            spliceCluster!(clusters[glob.rowcluster[n]],n,glob)
            pushCluster!(newclust,n,glob)
        end
    end
    push!(clusters,newclust)
end


function iterate_EM_BIC!(cl0::Cluster,glob::Params)
    llr = clusterLikelihood(glob,cl0)
    bic = get_bic(glob, llr, [cl0])
    if glob.debug
        println("iterate_EM_BIC: #clust=", 1, " llr=",llr," bic=",bic)
    end
    llr_best = llr
    bic_best = bic
    bic_last = 10000000000.0
    id_best = save_clusters(glob,[cl0])
    llr_last = llr
    clusters = [cl0]
    while (glob.nclust==0 && bic < bic_last) || (glob.nclust > 0 && length(clusters) < glob.nclust)
        if glob.debug
            println("iterate_EM_BIC: #clust=", length(clusters), " llr=",llr," bic=",bic)
        end
        bic_last = bic
        llr_last = llr
        if glob.nclust > 0 && false
            getNclusters!(clusters,glob)
        else
            addNewCluster!(clusters,glob)
        end
        optimizeClusters_EM!(glob,clusters)
        llr = sum([clusterLikelihood(glob,c) for c in clusters])
        bic = get_bic(glob, llr, clusters)
        if glob.debug
            println("iterate_EM_BIC: aftersplit #clust=", length(clusters), " llr=",llr," bic=",bic)
        end
        if (bic > bic_last  && glob.nclust==0) || any(map(x->x.nrows==0,clusters))
            # restore and return
            restore_clusters!(glob,clusters,id_best)
            return clusters
        else
            id_best = save_clusters(glob,clusters)
        end
    end
    return clusters
end




# find the average <P(D|C)^{β-1}> required for ML calculation

function find_Z(glob::Params, clusters::Vector{Cluster})
    # find average of P(D|C)^{β-1}
    ll_list = Float64[]
    ll_best = -10000000000.0
    clusterids = save_clusters(glob,clusters)
    nrows = glob.nrows
    Niter1 = glob.niter
    Niter = Niter1 * nrows
    nsnap = Niter ÷ 1000
    if nsnap==0
        nsnap = 1
    end
    nC = length(clusters)

    scores = Float64[0.0 for i in 1:nC]

    ll_running = sum([clusterLikelihood(glob,c) for c in clusters])

    for n = 1:Niter # equilibriation = 10%
        nr = rand(1:nrows)
        nclust = glob.rowcluster[nr]
        spliceCluster!(clusters[nclust],nr,glob)
        cl_orig = nclust
        
        # Gibbs select over all clusters
        
        for nc1 in 1:nC
            scores[nc1] = rowLikelihood(nr,clusters[nc1],glob)
        end
        # select from exp(scores)
        newclust = pickFromScores(scores)
        pushCluster!(clusters[newclust],nr,glob)
        ll_running += scores[newclust]-scores[nclust]
        if mod(n,nsnap)==0 && n>Niter÷10 # equilibriation = 10%
            push!(ll_list, ll_running)
        end
    end
    restore_clusters!(glob,clusters,clusterids)
    lZ = avg_logliklist(ll_list*(glob.beta-1))
    return lZ
end


function sampleClusters_HM(glob::Params, clusters::Vector{Cluster})

    ll_list = Float64[]
    clusterids = save_clusters(glob,clusters)
    nrows = glob.nrows
    counts = [Dict{Int64,Float64}() for c in clusters]
    
    Niter1 = glob.niter
    Niter = Niter1 * nrows
    nsnap = Niter ÷ 1000
    if nsnap == 0
        nsnap = 1
    end

    nC = length(clusters)
    scores = Float64[0.0 for i in 1:nC]

    for count in counts
        for n in 1:glob.nrows
            count[n] = 0.0
        end
    end
     
    ll_running = sum([clusterLikelihood(glob,c) for c in clusters])
    ll_best = ll_running
    if glob.debug
        println("sampleclustersHM 1 ", [(c.clustid, c.nrows) for c in clusters])
        println("ll_running= ",ll_running)
    end
    for n = 1:Niter # equilibriation = 10%
        nr = rand(1:nrows)
        nclust = glob.rowcluster[nr]
        # don't leave an empty cluster
        if clusters[nclust].nrows==1
            continue
        end
        spliceCluster!(clusters[nclust],nr,glob)
        # Gibbs select over all clusters
        for nc1 in 1:nC
            scores[nc1] = rowLikelihood(nr,clusters[nc1],glob)
        end
        # select from exp(scores)
        newclust = pickFromScores(scores*glob.beta)
        pushCluster!(clusters[newclust],nr,glob)
        ll_running += (scores[newclust]-scores[nclust])

        
        if mod(n,nsnap)==0 && n>Niter÷10 # equilibriation = 10%
            # FIXME if we even want counts...
            for r in 1:glob.nrows
                counts[glob.rowcluster[r]][r] += 1.0/Niter1
            end
            push!(ll_list, ll_running)
            if ll_running > ll_best
                ll_best = ll_running
                clusterids = save_clusters(glob,clusters)
            end
        end
    end

    if glob.debug
        println("sampleclustersHM 2 ", [(c.clustid, c.nrows) for c in clusters])
    end
    restore_clusters!(glob,clusters,clusterids)
    if glob.debug
        println("sampleclustersHM 3 ", [(c.clustid, c.nrows) for c in clusters])
    end
    lZ = find_Z(glob,clusters)
    ll_hm = -avg_logliklist(-ll_list*glob.beta) -lZ
    # ll_hm = -1/glob.beta * avg_logliklist(-ll_list*glob.beta)
    #ll_hm = -1/glob.beta * avg_logliklist(-ll_list*glob.beta) + 1/glob.beta*(1-glob.beta)*nrows*log(length(clusters))
    return counts, ll_hm, ll_best, clusterids
end


# simpson's rule, assume x is equally spaced
function simpson(x::Vector{Float64},y::Vector{Float64})
    h = x[2]-x[1]
    n = convert(Int64,round((x[end]-x[1])/h))
    integ = y[1]+y[end]
    for i = 1:n÷2
        integ += 4*y[2*i]
    end
    for i = 1:(n÷2-1)
        integ += 2*y[2*i+1]
    end
    return h/3*integ
end
                   

# thermodynamic integration:
# log ML = int_0^1 <log(P(D|C,θ)>_β dβ
# where <X>_β = \int X q_β(θ)/Z_β dθ
# and q_β(θ) = P(D|C,θ)^β P(θ|C) ≈ P(D|(C,θ)^β
# So, at a given β, sample from P(D|C,θ)^β and average log P(D|C,θ)
# Integrate over β using Simpson's rule

# in sampling, generally: try
# calc total cluster likelihood once at beginning
# update with -rowlikelihood(oldcluster)+rowlikelihood(newcluster)


function sampleClusters_TI(glob::Params, clusters::Vector{Cluster})

    clusterids = save_clusters(glob,clusters)
    integrand = Float64[]
    betalist = Float64[]
    nrows = glob.nrows
    
    for beta = 0.0:0.1:1.0
        ll_list = Float64[]
        
        Niter1 = glob.niter
        Niter = Niter1 * nrows
        nsnap = Niter ÷ 1000
        if nsnap==0
            nsnap = 1
        end
        nC = length(clusters)
        scores = Float64[0.0 for i in 1:nC]

        ll_running = sum([clusterLikelihood(glob,c) for c in clusters])
        for n = 1:Niter # equilibriation = 10%
            nr = rand(1:nrows)
            nclust = glob.rowcluster[nr]
            # don't leave an empty cluster
            if clusters[nclust].nrows==1
                continue
            end
            spliceCluster!(clusters[nclust],nr,glob)
            cl = clusters[nclust]
            #= if cl.nrows != length([x for x in cl.rows if x])
            error("samplecluster_ti spliceCluster: nrows=",cl.nrows," but ", length([x for x in cl.rows if x])," true")
            end 
            =#
            # Gibbs select over all clusters
            for nc1 in 1:nC
                scores[nc1] = rowLikelihood(nr,clusters[nc1],glob)
            end
            # select from exp(scores)
            newclust = pickFromScores(scores*beta)
            pushCluster!(clusters[newclust],nr,glob)
            ll_running += (scores[newclust]-scores[nclust])
            #=
            cl = clusters[newclust]            
            if cl.nrows != length([x for x in cl.rows if x])
            error("samplecluster_ti pushCluster: nrows=",cl.nrows," but ", length([x for x in cl.rows if x])," true")
            end
            =#
            
            if mod(n,nsnap)==0 && n>Niter÷10 # equilibriation = 10%
            #if n > Niter÷10
                #ll = sum([clusterLikelihood(glob,c) for c in clusters])        
                push!(ll_list, ll_running)
            end
        end
        #ll = sum([clusterLikelihood(glob,c) for c in clusters])                
        #println(ll," ",ll_running)

        restore_clusters!(glob,clusters,clusterids)
        #ll = avg_logliklist(ll_list)
        ll = sum(ll_list)/length(ll_list)
        push!(betalist,beta)
        push!(integrand,ll)
    end
    # ll_hm = -1/glob.beta * avg_logliklist(-ll_list*glob.beta)
    #ll_hm = -1/glob.beta * avg_logliklist(-ll_list*glob.beta) + 1/glob.beta*(1-glob.beta)*nrows*log(length(clusters))
    return simpson(betalist,integrand)
end


function iterate_EM_HM!(cl0::Cluster,glob::Params)
    llr = clusterLikelihood(glob,cl0)
    ll_hm = llr
    if glob.debug
        println("iterate_EM_HM #clust=", 1, " llr=",llr," ll_hm=",ll_hm)
    end
    ll_hm_best = ll_hm
    ll_hm_last = -10000000000.0
    id_best = save_clusters(glob,[cl0])
    llr_last = llr
    clusters = [cl0]
    while (glob.nclust == 0 && ll_hm > ll_hm_last) || (glob.nclust > 0 && length(clusters) < glob.nclust)
        if glob.debug
            println("iterate_EM_HM #clust=", length(clusters), " llr=",llr," ll_hm=",ll_hm)
        end
        ll_hm_last = ll_hm
        llr_last = llr
        #=
        llr_best = llr
        id = save_clusters(glob,clusters)
        id_best = save_clusters(glob,clusters)
        for c = 1:length(clusters)
            addNewCluster_one!(clusters,glob,c)
            optimizeClusters_EM!(glob,clusters)
            llr = sum([clusterLikelihood(glob,c) for c in clusters])
            println("c=",c," llr=",llr)
            if llr > llr_best
                llr_best = llr
                id_best = save_clusters(glob,clusters)
            end
            restore_clusters!(glob,clusters,id)
        end
        addNewCluster_one!(clusters,glob,1)
        restore_clusters!(glob,clusters,id_best)
        #id = save_clusters(glob,clusters)
        id = id_best
        =#
        addNewCluster!(clusters,glob)
        optimizeClusters_EM!(glob,clusters)
        llr = sum([clusterLikelihood(glob,c) for c in clusters])
        id = save_clusters(glob,clusters)        
        counts,ll_hm,ll_best,clusterids = sampleClusters_HM(glob,clusters)
        restore_clusters!(glob,clusters,id)
        if glob.debug
            println("iterate_EM_HM aftersample #clust=", length(clusters), " llr=",llr," llbest=",ll_best," ll_hm=",ll_hm)
        end
        flush(stdout)
        if (ll_hm < ll_hm_last  && glob.nclust==0) || any(map(x->x.nrows==0,clusters))
            # restore and return
            restore_clusters!(glob,clusters,id_best)
            if glob.debug
                println("iterate_EM_HM afterRestore #clust=", length(clusters), " llr=",llr_last," ll_hm=",ll_hm_last)
            end
            return clusters
        else
            id_best = save_clusters(glob,clusters)
        end
    end
    return clusters    
end


function iterate_EM_TI!(cl0::Cluster,glob::Params)
    llr = clusterLikelihood(glob,cl0)
    ll_TI = llr
    if glob.debug
        println("iterate_EM_TI #clust=", 1, " llr=",llr," ll_TI=",ll_TI)
    end
    ll_TI_best = ll_TI
    ll_TI_last = -10000000000.0
    id_best = save_clusters(glob,[cl0])
    llr_last = llr
    clusters = [cl0]
    while (glob.nclust==0 && ll_TI > ll_TI_last) || (glob.nclust > 0 && length(clusters) < glob.nclust)
        if glob.debug
            println("iterate_EM_TI #clust=", length(clusters), " llr=",llr," ll_TI=",ll_TI)
        end
        ll_TI_last = ll_TI
        llr_last = llr
        addNewCluster!(clusters,glob)
        optimizeClusters_EM!(glob,clusters)
        llr = sum([clusterLikelihood(glob,c) for c in clusters])
        id = save_clusters(glob,clusters)
        ll_TI = sampleClusters_TI(glob,clusters)
        #restore_clusters!(glob,clusters,id)
        #println("aftersample #clust=", length(clusters), " llr=",llr," ll_hm=",ll_hm)
        flush(stdout)
        if (ll_TI < ll_TI_last  && glob.nclust==0) || any(map(x->x.nrows==0,clusters))
            # restore and return
            restore_clusters!(glob,clusters,id_best)
            if glob.debug
                println("iterate_EM_TI afterRestore #clust=", length(clusters), " llr=",llr_last," ll_TI=",ll_TI_last)
            end
            return clusters
        else
            id_best = save_clusters(glob,clusters)
        end
    end
    return clusters    
end




# used for calculating rand score
function clustnums(clusters,glob)
    clnums::Vector{Vector{Int64}} = [[] for x in clusters]
    for i = 1:length(clusters)
        c = clusters[i]
        for n in 1:glob.nrows
            if c.rows[n]
                push!(clnums[i],n)
            end
        end
    end
    return [Set(c) for c in clnums]
    #return [Set([sc.rows[n].rowind for n in 1:sc.nrows]) for sc in cluster]
    #FIXME
end

function choose(n,m)
    if n<m
        return 0.0
    else
        return exp(lgamma(n+1) - lgamma(m+1) -lgamma(n-m+1))
    end
end

function choose2(n)
    return n*(n-1)/2
end

# adjusted rand index of two clusters
function randscore(clust1::Vector{Set{Int64}},clust2::Vector{Set{Int64}})
    ntot = sum([length(c) for c in clust1])
    l1 = length(clust1)
    l2 = length(clust2)
    nmat = zeros(Float64,(l1,l2))
    for n in 1:l1
        for m in 1:l2
            nmat[n,m] = length(intersect(clust1[n],clust2[m]))
        end
    end
    avec = zeros(Float64,l2)
    bvec = zeros(Float64,l1)
    for n in 1:l1
        bvec[n] = sum(nmat[n,:])
    end
    for n in 1:l2
        avec[n] = sum(nmat[:,n])
    end
    index = sum([choose(x,2) for x in nmat])
    maxindex = 0.5*(sum([choose(a,2) for a in avec]) + sum([choose(b,2) for b in bvec]))
    expindex = (sum([choose(a,2) for a in avec])*sum([choose(b,2) for b in bvec]))/choose(ntot,2)
    return index/maxindex, (index-expindex)/(maxindex-expindex)
end

function print_params(glob::Params, criterion::String, filename::String, suffix::String)
    println("−"^50)
    println(" "^20 * "Parameters")
    println("−"^50)
    println("c (pseudocount)                       : ",glob.c)
    println("μ₀ (normal-gamma param)               : ",glob.μ₀)
    println("β₀ (normal-gamma param)               : ",glob.β₀)
    println("a₀ (normal-gamma param)               : ",glob.a₀)
    println("b₀ (normal-gamma param)               : ",glob.b₀)
    println("lrow_c (number of categorical columns): ",glob.lrow_c)
    println("lrow_n (number of numeric columns)    : ",glob.lrow_n)
    println("nrows (total number of rows)          : ",glob.nrows)
    println("maxnclust (max clusters, 0=no limit)  : ",glob.maxnclust)
    println("nclust (fixed #clusters, 0= not fixed): ",glob.nclust)
    if criterion=="B"
        println("model selection criterion             : BIC")
    elseif criterion=="H"
        println("model selection criterion             : Marginal likelihood (HMβ)")
        println("beta                                  : ", glob.beta)
        println("niter                                 : ", glob.niter)
    else
        println("model selection criterion             : Marginal likelihood (TI)")
        println("niter                                 : ", glob.niter)
    end
    println("")
    println("Input filename                        : ",filename)    
    println("Output filename                       : ",filename*suffix)    
    println("−"^50)    
end

function print_output(filename::String,clusters::Vector{Cluster},glob::Params)
    f = open(filename,"w")
    for n in 1:glob.nrows
        for c in 1:length(clusters)
            if clusters[c].rows[n]
                write(f,"$c\n")
                break
            end
        end
    end
    close(f)
end
    
    
