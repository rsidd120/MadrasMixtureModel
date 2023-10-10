#!/usr/bin/env julia

# (C) Rahul Siddharthan, 2023
# generate ynthetic tabular datasets consisting of categorical and numeric columns
# run with -h for options

using Random;
using SpecialFunctions;
using Distributions;
using ArgParse;


function pick_mean_stdev(lower,upper)
    m = lower + rand()*(upper-lower)
    stdv = abs(rand()*(upper-lower))
    return m,stdv
end

function getDist(n,c,upper,lower)
    if n==0
        m,s= pick_mean_stdev(lower,upper)
        return Normal(m,s)
    else
        return Categorical(rand(Dirichlet(n,c)))
    end
end

function getDistMSD(n,c,m,s)
    if n==0
        return Normal(m,s)
    else
        return Categorical(rand(Dirichlet(n,c)))
    end
end

function sampleFromProbVec(x)
    n = length(x)
    r = rand()
    m = 1
    t = x[1]
    while m < n && t < r
        m += 1
        t += x[m]
    end
    return m
end

getSequence(typevec,dists) = 
    map(n-> rand(dists[n]), 1:length(typevec))

    
function makeClusterMSD(N,typevec,c,meanvec,stdvec)
    dists = [getDistMSD(typevec[n],c,meanvec[n],stdvec[n]) for n in 1:length(typevec)]
    seqs = []
    for n=1:N
        row1 = getSequence(typevec,dists)
        push!(seqs,row1)
    end
    return seqs
end

function get_typevec(dtstr::String)
    types = split(dtstr,",")
    typevec::Vector{Int64} = []
    for t in types
        ts = split(t,":")
        if ts[1]=="n"
            t0 = 0
        elseif ts[1]=="g"
            t0 = -1
        else
            t0 = parse(Int64,ts[1])
        end
        nt = parse(Int64,ts[2])
        for n = 1:nt
            push!(typevec,t0)
        end
    end
    return typevec
end

function get_range(rng::String)
    rngs = split(rng,":")
    return parse(Float64,rngs[1]), parse(Float64,rngs[2])
end

function get_clustsizes(sizes::String, nrows::Int64)
    sizess = split(sizes,",")
    sizesf = [parse(Float64,x) for x in sizess]
    sizesf = sizesf/sum(sizesf)*nrows
    sizesi = [Int(round(x)) for x in sizesf]
    while sum(sizesi) < nrows
        sizesi[end] += 1
    end
    while sum(sizesi) > nrows
        sizesi[end] -= 1
    end
    return sizesi
end

function parse_cmdline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--alpha_range","-a"
        help = "range of values to pick α from, eg 0.5:2.0 means in that range"
        arg_type = String        
        default = "0:1"
        "--theta_range","-Q"
        help = "range of values to pick θ from, eg 0.5:2.0 means in that range"
        arg_type = String        
        default = "0:1"        
        "--targetvar","-t"
        help = "should a target variable be output (last column), and if so, binary (B) or float (F), Default: no"
        arg_type = String
        default = ""
        "--targetnoise", "-T"
        help = "noise in dependent variable, as fraction of range of variable (default: 0.1 = 10%)"
        arg_type = Float64
        default=0.1
        "--datatypes", "-d"
        help = "how many of each normal / gamma / categorical variable; format = \"n:3,g:2,2:4,4:5\" means 3 normal, 2 gamma, 4 2-valued, 5 4-valued\""
        arg_type = String
        default = "0:100"
        "--mean_diff","-m"
        help = "difference between means for different clusters"
        arg_type = Float64
        default = 1.0
        "--std_sep","-s"
        help = "range of values to pick std deviation from, eg 0.5:2.0 means in that range"
        arg_type = String        
        default = "0:1"
        "--std_diff","-S"
        help = "stdev increment between clusters, default 0 = ignored"
        arg_type = Float64
        default = 0.0
        "--pseudocount", "-c"
        help = "Dirichlet pseudocount for categorical data"
        arg_type = Float64
        default = 0.5
        "--categorical_difference", "-D"
        help = "Difference between categorical vectors across clusters. For each cluster, v = (v0 + Dv1)/(1+D) where v0 is common and v1 is specific to cluster. D=0.01 means very similar, D=100 means very different."
        arg_type = Float64
        default = 1.0
        "--nrows", "-N"
        arg_type = Int64
        default = 1000
        help = "Number of data rows to output"
        "--clustsizes", "-C"
        arg_type = String
        default = "1,1,1,1"
        help = "Relative sizes of clusters, eg \"9,1\" means two clusters in 9:1 ratio"
        "--ignore", "-i"
        arg_type = Int64
        default = 0
        help = "Number of columns to be ignored (uninformative columns)"
        "filename"
        help = "output base filename, .csv for data, .labels for labels"
        arg_type = String
        required = true
    end
    return parse_args(s)
end

function get_data(typevec,meandiff, stddiff, σ1, σ2, c, cat_diff, α1, α2, θ1, θ2, nrows, nignore, clustsizes)
    maxcat = maximum(typevec)

    # categorical variables are sampled from (v1 + cat_diff*v2)/(1+cat_diff)
    # if cat_diff is small, categorical columns are similar
    
    v1_base = []
    for n = 1:length(typevec)
        if typevec[n]==0 || typevec[n]== -1
            push!(v1_base,[0.0])
        else
            v1 = rand(Dirichlet(typevec[n],c))
            push!(v1_base, v1)
        end
    end

    # uninformative positions
    ignorelist = sample(1:length(typevec),nignore; replace=false)

    distvec0 = []
    for n in 1:length(typevec)
        if typevec[n]==0
            δσ = σ1 + rand()*(σ2-σ1)
            µ = 0
            σ = δσ 
            push!(distvec0,Normal(μ,σ))
        elseif typevec[n]== -1
            α = α1 + rand()*(α2-α1)
            θ = θ1 + rand()*(θ2-θ1)
            push!(distvec0,Gamma(α,θ))
        else
            v2 = rand(Dirichlet(typevec[n],c))
            v = (v1_base[n] + cat_diff*v2)/(1.0+cat_diff)
            push!(distvec0,Categorical(v))
        end
    end
    
    data = []
    for k in 1:length(clustsizes)
        csize = clustsizes[k]
        distvec = []
        for n in 1:length(typevec)
            if n in ignorelist
                push!(distvec,distvec0[n])
            elseif typevec[n]==0
                if stddiff > 0.0
                    σ = σ1 + stddiff*(k-1)
                else
                    δσ = σ1 + rand()*(σ2-σ1)
                    σ =  δσ
                end
                µ = meandiff*(k-1)
                push!(distvec,Normal(μ,σ))
            elseif typevec[n]== -1
                α = α1 + rand()*(α2-α1)
                θ = θ1 + rand()*(θ2-θ1)
                push!(distvec,Gamma(α,θ))                
            else
                v2 = rand(Dirichlet(typevec[n],c))
                v = (v1_base[n] + cat_diff*v2)/(1.0+cat_diff)
                push!(distvec,Categorical(v))
            end
        end
        for m = 1:csize
            push!(data, (k, [rand(distvec[n]) for n = 1:length(typevec)]))
        end
    end
    shuffle!(data)
    return data
end

function bindep(dep,depthres)
    if dep > depthres
        return 1
    else
        return 0
    end
end

function get_dep(data,depvar,depnoise)
    coeffs = [2*rand()-1 for x in 1:length(data[1][2])]
    deps = [sum(coeffs .* x[2]) for x in data]
    depmin = minimum(deps)
    depmax = maximum(deps)
    depnoiserange = depnoise*(depmax-depmin)
    deps = [x + depnoiserange*rand() for x in deps]
    if depvar=="B"
        depthres = sort(deps)[length(deps)÷2]
        deps = [bindep(dep,depthres) for dep in deps]
    end
    for n in 1:length(deps)
        push!(data[n][2],deps[n])
    end
    return data
end

function main()
    typestr=""
    meandiff = 1.0
    stddiff = 0.0
    alpharange = ""
    thetarange = ""
    stdsep = ""
    c = 0.0
    cat_diff = 0.0
    nrows = 0
    nignore=0
    cluststr = ""
    filename = ""
    depvar = ""
    depnoise = 0.1
    parsed_args = parse_cmdline()
    for (arg,val) in parsed_args
        if arg=="datatypes"
            typestr = val
        elseif arg=="targetvar"
            depvar = val
        elseif arg=="targetnoise"
            depnoise = val
        elseif arg=="alpha_range"
            alpharange = val
        elseif arg=="theta_range"
            thetarange = val
        elseif arg=="mean_diff"
            meandiff = val
        elseif arg=="std_diff"
            stddiff = val
        elseif arg=="std_sep"
            stdsep = val
        elseif arg=="pseudocount"
            c = val
        elseif arg=="categorical_difference"
            cat_diff = val
        elseif arg=="nrows"
            nrows = val
        elseif arg=="clustsizes"
            cluststr = val
        elseif arg=="ignore"
            nignore = val
        elseif arg=="filename"
            filename=val
        end
    end

    α1, α2 = get_range(alpharange)
    θ1, θ2 = get_range(thetarange)    
    typevec = get_typevec(typestr)
    σ1, σ2 = get_range(stdsep)
    clustsizes = get_clustsizes(cluststr,nrows)

    data = get_data(typevec,meandiff, stddiff, σ1, σ2, c, cat_diff, α1, α2, θ1, θ2, nrows, nignore, clustsizes)
    if depvar == "B" || depvar == "F"
        data = get_dep(data, depvar, depnoise)
    end
    fcsv = open(filename*".csv","w")
    flabels = open(filename*".labels","w")
    
    for (k,row) in data
        write(flabels,string(k)*"\n")
        write(fcsv,join([string(x) for x in row],",")*"\n")
    end    
end

main()
