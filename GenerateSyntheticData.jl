# Functions for generating synthetic data from a clustering
# (C) 2023 Chandrani Kumari


# using Suppressor: @suppress_err
# @suppress_err using BloomFilters

using SpecialFunctions
using Distributions
using DataFrames
using Statistics
using StatsBase
using ArgParse
using Random
using CSV
using GLM
using StatsPlots
using MLBase



function get_typevec(dtstr::String)
    types = split(dtstr,",")
    typevec::Vector{Int64} = []
    for t in types
        ts = split(t,":")
        t0 = parse(Int64,ts[1])
        nt = parse(Int64,ts[2])
        for n = 1:nt
            push!(typevec,t0)
        end
    end
    return typevec
end

function mean_std(df::DataFrame, continousC::Vector{String})
    return mean.(eachcol(df[!, continousC])),std.(eachcol(df[!, continousC]))
end


function get_prob_vector(df::DataFrame, categoricalC::Vector{String})
    prob_vector =[]
    for c in categoricalC
        push!(prob_vector,values(sort(countmap(df[!,c])))./nrow(df))
    end
    return prob_vector
end

function get_clusterlabel(filename::String)
    # read file contents, one line at a time
    clusterLabel = []
    open(filename)  do f
      # read till end of file
      while ! eof(f) 
         # read a new / next line for every iteration          
         s = readline(f)         
         #line += 1
         push!(clusterLabel,parse(Int64,s))
      end
    end
    return clusterLabel
end

function get_data(typevec, µ, σ, prob_vector, csize) 
    
    distvec = []
    i, j = 1, 1
    for n in 1:length(typevec)
        if typevec[n]==0
            push!(distvec,Normal(μ[i],σ[i]))
            i = i+1
        else
            push!(distvec,Categorical(prob_vector[j]))
            j = j+1
        end
    end
    data = []
    for m = 1:csize
        push!(data, [rand(distvec[n]) for n = 1:length(typevec)])
    end
    shuffle!(data)
    return data
end

function myDf(x,col)
  return rename!(convert(DataFrame, x), col)
end

function generateSynData(N::Int64, data::DataFrame, typevec::Vector{Int64}, categoricalC::Vector{String}, 
        continousC::Vector{String}, clusterLabel::Vector{Int64})
    
    synthetic_data = []
    for c in unique!(clusterLabel)
        cluster = select!(data[(data.Clabel .== c), :], Not([:Clabel]));
        
        μ , σ = mean_std(cluster, continousC)
        prob_vector = get_prob_vector(cluster, categoricalC)
        push!(synthetic_data, get_data(typevec, µ, σ, prob_vector, nrow(cluster)))
    end
    
    
    return synthetic_data
end

