#!/usr/bin/env julia
# (c) Rahul Siddharthan, 2023
# Licence: MIT
# Run with -h for options

using Random
using Distributions
using SpecialFunctions
using ArgParse;
include("mmm_lib.jl")

function parse_cmdline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--pseudocount", "-c"
        help = "Dirichlet prior pseudocount for categorical data"
        arg_type = Float64
        default = 0.5
        "--normalgamma", "-g"
        help = "normal gamma parameters μ₀, β₀, a₀, b₀ comma-separated (default 0.0,0.5,0.5,0.5)"
        arg_type = String
        default="0.0,0.5,0.5,0.5"
        "--minclustsize", "-m"
        help = "minimum cluster size below which it won't be split; default=20"
        arg_type = Int64
        default = 20
        "--maxnclust", "-M"
        help = "maximum number of output clusters; default=0 ≡ no limit"
        arg_type = Int64
        default = 0
        "--nclust", "-n"
        help = "split into exactly n clusters; default=0 (choose optimal n using BIC or TI or HMβ)"
        arg_type = Int64
        default = 0
        "--beta", "-b"
        help = "parameter β (inverse temperature) in HMβ method; default=0.5"
        arg_type = Float64
        default = 0.5
        "--niter", "-i"
        help = "number of passes through the full set when sampling, in TI or HMβ (default=10)"
        arg_type = Int64
        default = 10
        "--quiet", "-q"
        help = "display no output at all; default = display minimal progress output"
        action=:store_true
        "--verbose", "-V"
        help = "display verbose (debugging) output; default = display minimal progress output"
        action=:store_true
        "--criterion", "-C"
        help = "Criterion for model selection (B = BIC, H = HMβ, T = TI) (default: T)"
        arg_type = String
        default = "T"
        "--excludelast", "-x"
        help = "exclude last column (dependent variable) in clustering data"
        action = :store_true
        "--typestring", "-t"
        help = "string indicating type of each column (c or C for categorical, n or N for normal, eg CCNCCNN"
        arg_type = String
        default = ""
        "--forcefloat","-F"
        help = "force treating all fields as float (numeric data)"
        action=:store_true
        "--suffix","-s"
        help = "output filename suffix (default: .labels.mmm)"
        arg_type = String
        default = ".labels.mmm"
        "filename"
        help = "data file name, .csv format, no header (or header must begin with # and will be ignored)"
        arg_type = String
        required = true
    end
    return parse_args(s)
end


function main()
    # dummy or default values, to be overridden by arguments or inferred
    pscount = 0.5
    typevec=""
    ngstr = "0.0,0.5,0.5,0.5"
    minclustsize = 20
    maxnclust = 0
    nclust = 0
    beta = 0.5
    niter = 10
    quiet = false
    debug = false
    criterion = "T"
    filename = ""
    excludelast = false
    typestring = ""
    forcefloat = false
    suffix = ".labels.mmm"
    parsed_args = parse_cmdline()
    for (arg,val) in parsed_args
        if arg=="pseudocount"
            pscount = val
        elseif arg=="normalgamma"
            ngstr = val
        elseif arg=="minclustsize"
            minclustsize = val
        elseif arg=="maxnclust"
            maxnclust = val
        elseif arg=="nclust"
            nclust = val
        elseif arg=="beta"
            beta = val
        elseif arg=="niter"
            niter = val
        elseif arg=="quiet"
            quiet = val
        elseif arg=="verbose"
            debug = val
        elseif arg=="criterion"
            criterion = val
        elseif arg=="filename"
            filename = val
        elseif arg=="excludelast"
            excludelast = val
        elseif arg=="typestring"
            typestring = val
        elseif arg=="suffix"
            suffix = val
        elseif arg=="forcefloat"
            forcefloat = true
        end
    end    

    glob, cl = init_default(filename,typestring,forcefloat,excludelast);
    if typevec != ""
        glob.typevec = typevec
    end
    glob.c = pscount
    ngstrn = [parse(Float64,x) for x in split(ngstr,",")]
    glob.μ₀ = ngstrn[1]
    glob.β₀ = ngstrn[2]
    glob.a₀ = ngstrn[3]
    glob.b₀ = ngstrn[4]
    glob.minclustsize = minclustsize
    glob.maxnclust = maxnclust
    glob.nclust = nclust
    glob.quiet = quiet
    glob.debug = debug
    glob.beta = beta
    glob.niter = niter

    if !glob.quiet
        print_params(glob,criterion, filename, suffix);
    end
    if criterion=="B"
        clusters = iterate_EM_BIC!(cl,glob);
    elseif criterion=="H"
        clusters = iterate_EM_HM!(cl,glob);
    else
        clusters = iterate_EM_TI!(cl,glob);
    end
    
    print_output(filename*suffix,clusters,glob)
end

main()
